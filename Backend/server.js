const express = require('express');
const multer = require('multer');
const dotenv = require('dotenv');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Rate limiting middleware
const rateLimit = require('express-rate-limit');
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 5, // limit each IP to 1 request per minute
  message: {
    success: false,
    error: 'server crashed.'
  }
});

// Configure multer for memory storage
const upload = multer({
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Please upload an image file (PNG, JPEG)'), false);
    }
  }
});

if (!process.env.API) {
  console.warn('API is not set. Set it in .env for the server to work.');
}
const genAI = new GoogleGenerativeAI(process.env.API || '', { apiVersion: 'v1' });
// Model resolution: try env override first, then known-good fallbacks
const MODEL_CANDIDATES = [
  (process.env.GEMINI_MODEL || '').trim(),
  // Prefer best extractor first
  'gemini-1.5-pro-latest',
  'gemini-1.5-pro',
  'gemini-1.5-flash-latest',
  'gemini-1.5-flash',
  'gemini-1.0-pro-vision',
  'gemini-pro-vision',
  'gemini-1.0-pro',
  'gemini-pro'
].filter(Boolean);

// Dynamically discover available models for this API key and cache result
const https = require('https');
let discoveredModelsCache = null; // array of model names in preference order
const fetchDiscoveredModels = () => new Promise((resolve) => {
  if (discoveredModelsCache) return resolve(discoveredModelsCache);
  const apiKey = process.env.API;
  if (!apiKey) return resolve([]);
  const url = `https://generativelanguage.googleapis.com/v1/models?key=${encodeURIComponent(apiKey)}`;
  https.get(url, (res) => {
    let body = '';
    res.on('data', (chunk) => body += chunk);
    res.on('end', () => {
      try {
        const json = JSON.parse(body);
        const models = Array.isArray(json.models) ? json.models : [];
        // Prefer models that support image input and generateContent
        const preferred = models
          .filter(m => Array.isArray(m.supportedGenerationMethods) && m.supportedGenerationMethods.includes('generateContent'))
          .filter(m => Array.isArray(m.inputTokenLimit) || true) // keep all; modality info not always present
          .map(m => m.name)
          .filter(Boolean);
        // API returns fully-qualified names like 'models/gemini-1.5-flash'; normalize to short name
        const normalized = preferred.map(n => n.replace(/^models\//, ''));
        discoveredModelsCache = normalized;
        console.log(`Discovered ${normalized.length} Gemini models for this key.`);
        resolve(discoveredModelsCache);
      } catch (e) {
        console.warn('Failed to parse models list:', e.message);
        resolve([]);
      }
    });
  }).on('error', (err) => {
    console.warn('Failed to fetch models list:', err.message);
    resolve([]);
  });
});

app.use((req, res, next) => {
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');
  res.setHeader('Surrogate-Control', 'no-store');
  next();
});

app.post('/api/process-prescription', limiter, upload.single('file'), async (req, res) => {
  try {
    const file = req.file;
    if (!file) return res.status(400).json({
      success: false,
      error: 'No prescription image uploaded.'
    });

    // Convert image buffer to base64 with quality optimization
    const imageBase64 = file.buffer.toString('base64');
    
    // Log image details for debugging
    console.log(`Processing image: ${file.originalname}, Size: ${file.size} bytes, Type: ${file.mimetype}`);

    // Initialize Gemini model using dynamic discovery + candidates
    let model = null;
    let selectedModel = null;
    const discovered = await fetchDiscoveredModels();
    const uniqueOrdered = Array.from(new Set([...(discovered || []), ...MODEL_CANDIDATES]));
    for (const candidate of uniqueOrdered) {
      try {
        model = genAI.getGenerativeModel({
          model: candidate,
          generationConfig: {
            responseMimeType: 'application/json',
            temperature: 0.2
          }
        });
        selectedModel = candidate;
        console.log(`Using Gemini model: ${candidate}`);
        break;
      } catch (err) {
        console.warn(`Model init failed for ${candidate}: ${err.message}`);
      }
    }
    if (!model) {
      // Final fallback to a common JSON-capable text model
      selectedModel = 'gemini-1.0-pro';
      model = genAI.getGenerativeModel({
        model: selectedModel,
        generationConfig: {
          responseMimeType: 'application/json',
          temperature: 0.2
        }
      });
      console.log(`Falling back to default model: ${selectedModel}`);
    }

    // Prepare a simpler, more reliable prompt for medicine extraction
    const prompt = `
      Analyze this prescription image and extract medicine information in JSON format.
      
      Look for:
      - Medicine names
      - Dosages (mg, ml, tablets, etc.)
      - How often to take (frequency)
      - Duration of treatment
      - Any special instructions
      - Doctor's name
      - Patient's name
      - Date
      
      Return ONLY valid JSON in this exact format:
      {
        "medicines": [
          {
            "name": "medicine name here",
            "dosage": "dosage here",
            "frequency": "how often to take",
            "duration": "duration here",
            "instructions": "any special instructions"
          }
        ],
        "doctor": {
          "name": "doctor name"
        },
        "patient": {
          "name": "patient name"
        },
        "date": "prescription date"
      }
      
      If you can't read something, use "not visible". Return ONLY the JSON, no other text.
    `;

    // Helpers to coerce LLM output into valid JSON reliably
    const tryParseJson = (text) => {
      try { return JSON.parse(text); } catch { return null; }
    };
    const extractLikelyJson = (text) => {
      if (!text) return '';
      // Remove code fences
      let cleaned = text.replace(/```json|```/gi, '').trim();
      // If already parses, use it
      if (tryParseJson(cleaned)) return cleaned;
      // Extract first {...} or [..]
      const objMatch = cleaned.match(/\{[\s\S]*\}/);
      const arrMatch = cleaned.match(/\[[\s\S]*\]/);
      const candidate = objMatch?.[0] || arrMatch?.[0] || cleaned;
      return candidate.trim();
    };
    const weakRepairJson = (text) => {
      if (!text) return text;
      let t = text
        // remove trailing commas before closing braces/brackets
        .replace(/,\s*([}\]])/g, '$1')
        // ensure keys are quoted (simple heuristic)
        .replace(/([,{\[]\s*)([A-Za-z0-9_]+)\s*:/g, '$1"$2":');
      return t;
    };

    // Generate content using Gemini with retry mechanism and better error handling
    let responseText = '';
    let data = null;
    let attempts = 0;
    const maxAttempts = 3;

    const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    while (attempts < maxAttempts && !data) {
      attempts++;
      console.log(`Attempt ${attempts} to process prescription...`);
      
      try {
        // Add delay between attempts to avoid rate limiting
        if (attempts > 1) {
          console.log(`Waiting 3 seconds before retry...`);
          await delay(3000);
        }

        const result = await model.generateContent([
          {
            inlineData: {
              mimeType: file.mimetype,
              data: imageBase64
            }
          },
          { text: prompt }
        ]);

        responseText = result.response.text();
        console.log(`Raw API Response (Attempt ${attempts}):`, responseText);

        // Prefer strict JSON first
        let candidate = extractLikelyJson(responseText);
        data = tryParseJson(candidate);
        if (!data) {
          // Attempt weak repair and parse again
          candidate = weakRepairJson(candidate);
          data = tryParseJson(candidate);
        }
        if (!data && candidate !== responseText) {
          // Last resort: try full text with weak repair
          const repairedFull = weakRepairJson(responseText);
          data = tryParseJson(extractLikelyJson(repairedFull));
        }
        if (!data) {
          throw new Error('Invalid JSON from model');
        }
        console.log(`Successfully parsed JSON on attempt ${attempts}`);
        
      } catch (error) {
        console.warn(`Error on attempt ${attempts}:`, error.message);
        // No API key rotation
        // If the model is not supported on this API version, try the next candidate
        if (/(404|not found|not supported)/i.test(error.message)) {
          const currentIndex = MODEL_CANDIDATES.indexOf(selectedModel);
          const nextModel = MODEL_CANDIDATES[currentIndex + 1];
          if (nextModel) {
            try {
              console.log(`Retrying with alternate model: ${nextModel}`);
              model = genAI.getGenerativeModel({
                model: nextModel,
                generationConfig: {
                  responseMimeType: 'application/json',
                  temperature: 0.2
                }
              });
              selectedModel = nextModel;
              continue; // redo the attempt with new model
            } catch (switchErr) {
              console.warn(`Failed to switch model to ${nextModel}: ${switchErr.message}`);
            }
          }
        }
        
        if (error.status === 503 || error.message.includes('overloaded') || error.message.includes('Service Unavailable')) {
          console.log(`Service overloaded, waiting 5 seconds before retry...`);
          await delay(5000);
        } else if (attempts < maxAttempts) {
          // Try with an even simpler prompt on subsequent attempts
          const fallbackPrompt = `
            Read this prescription and return JSON:
            {
              "medicines": [{"name": "medicine name", "dosage": "dosage", "frequency": "frequency"}],
              "doctor": {"name": "doctor name"},
              "patient": {"name": "patient name"},
              "date": "date"
            }
            Use "not visible" if you can't read something.
          `;
          
          try {
            const fallbackResult = await model.generateContent([
              {
                inlineData: {
                  mimeType: file.mimetype,
                  data: imageBase64
                }
              },
              { text: fallbackPrompt }
            ]);

            responseText = fallbackResult.response.text();
            let candidate = extractLikelyJson(responseText);
            data = tryParseJson(candidate) || tryParseJson(weakRepairJson(candidate));
            if (!data) throw new Error('Invalid JSON from fallback');
            console.log(`Successfully parsed JSON on fallback attempt`);
          } catch (fallbackError) {
            console.error(`Fallback attempt also failed:`, fallbackError.message);
          }
        }
      }
    }

    if (data) {
      return res.json({
        success: true,
        data: data,
        error: null,
        attempts: attempts,
        model: selectedModel
      });
    } else {
      return res.json({
        success: false,
        data: null,
        error: "Error",
        rawResponse: responseText,
        attempts: attempts
      });
    }
  } catch (error) {
    console.error('Error processing prescription:', error);

    // Handle specific error types
    if (error.status === 503 || error.message.includes('overloaded') || error.message.includes('Service Unavailable')) {
      return res.status(503).json({
        success: false,
        data: null,
        error: 'Gemini AI service is temporarily overloaded. Please try again in a few minutes.',
        retryAfter: 60
      });
    } else if (error.status === 429) {
      return res.status(429).json({
        success: false,
        data: null,
        error: 'Rate limit exceeded. Please wait before trying again.'
      });
    } else if (error.message.includes('API key')) {
      return res.status(401).json({
        success: false,
        data: null,
        error: 'Invalid API key. Please check your Gemini API configuration.'
      });
    } else {
      return res.status(500).json({
        success: false,
        data: null,
        error: 'An error occurred while processing the prescription. Please try again.'
      });
    }
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
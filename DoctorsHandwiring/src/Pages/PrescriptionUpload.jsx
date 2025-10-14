import React, { useState } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { Button, Typography, Box, CircularProgress, Paper, List, ListItem, ListItemText, Divider, Tabs, Tab } from '@mui/material';
import InvoiceGenerator from '../Components/InvoiceGenerator';


const PrescriptionUpload = () => {
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [activeTab, setActiveTab] = useState(0);

  const onDrop = (acceptedFiles) => {
    setFile(acceptedFiles[0]);
    setErrorMessage('');
    setResponse(null); // Clear previous response
  };

  const handleUpload = async () => {
    if (!file) {
      setErrorMessage('Please select a file to upload.');
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setErrorMessage('');

    try {
      const res = await axios.post('http://localhost:5000/api/process-prescription', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (res.data.success) {
        setResponse(res.data.data);
      } else {
        setErrorMessage(res.data.error || 'Error');
      }
    } catch (error) {
      console.error("Error uploading file", error);
      setErrorMessage(error.response?.data?.error || 'Error');
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps } = useDropzone({
    accept: 'image/jpeg, image/png, image/jpg',
    onDrop,
    multiple: false,
  });

  const renderPrescriptionData = () => {
    if (!response) return null;

    return (
      <Box sx={{ marginTop: 3, width: '100%' }}>
        <Typography 
          variant="h5" 
          gutterBottom 
          sx={{ 
            color: 'white',
            fontWeight: 600,
            mb: 3,
            textAlign: 'center'
          }}
        >
          📋 Prescription Analysis Results
        </Typography>

        {/* Doctor Information */}
        {response.doctor && (
          <Paper sx={{ 
            padding: 3, 
            marginBottom: 3,
            background: 'rgba(255, 255, 255, 0.15)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: 2
          }}>
            <Typography variant="h6" fontWeight="bold" sx={{ color: 'black', mb: 2 }}>
              👨‍⚕️ Doctor Information
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText 
                  primary="Name" 
                  secondary={response.doctor.name || 'Not specified'}
                  primaryTypographyProps={{ color: 'rgba(0, 0, 0, 0.6)' }}
                  secondaryTypographyProps={{ color: 'black', fontWeight: 500 }}
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Specialization" 
                  secondary={response.doctor.specialization || 'Not specified'}
                  primaryTypographyProps={{ color: 'rgba(0, 0, 0, 0.6)' }}
                  secondaryTypographyProps={{ color: 'black', fontWeight: 500 }}
                />
              </ListItem>
            </List>
          </Paper>
        )}

        {/* Patient Information */}
        {response.patient && (
          <Paper sx={{ 
            padding: 3, 
            marginBottom: 3,
            background: 'rgba(255, 255, 255, 0.15)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: 2
          }}>
            <Typography variant="h6" fontWeight="bold" sx={{ color: 'black', mb: 2 }}>
              👤 Patient Information
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText 
                  primary="Name" 
                  secondary={response.patient.name || 'Not specified'}
                  primaryTypographyProps={{ color: 'rgba(0, 0, 0, 0.6)' }}
                  secondaryTypographyProps={{ color: 'black', fontWeight: 500 }}
                />
              </ListItem>
              <ListItem>
                <ListItemText 
                  primary="Age" 
                  secondary={response.patient.age || 'Not specified'}
                  primaryTypographyProps={{ color: 'rgba(0, 0, 0, 0.6)' }}
                  secondaryTypographyProps={{ color: 'black', fontWeight: 500 }}
                />
              </ListItem>
            </List>
          </Paper>
        )}

        {/* Medicines List */}
        {response.medicines && response.medicines.length > 0 && (
          <Paper sx={{ 
            padding: 3, 
            marginBottom: 3,
            background: 'rgba(255, 255, 255, 0.15)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: 2
          }}>
            <Typography variant="h6" fontWeight="bold" sx={{ color: 'black', mb: 2 }}>
              💊 Prescribed Medicines ({response.medicines.length})
            </Typography>
            <List>
              {response.medicines.map((medicine, index) => (
                <React.Fragment key={index}>
                  <ListItem alignItems="flex-start" sx={{ 
                    background: 'rgba(255, 255, 255, 0.1)',
                    borderRadius: 2,
                    mb: 1,
                    border: '1px solid rgba(255, 255, 255, 0.1)'
                  }}>
                    <ListItemText
                      primary={
                        <Typography variant="h6" sx={{ color: 'black', fontWeight: 600 }}>
                          {medicine.name || 'Medicine name not specified'}
                        </Typography>
                      }
                      secondary={
                        <Box sx={{ mt: 1 }}>
                          <Typography component="span" variant="body2" sx={{ color: 'rgba(0, 0, 0, 0.8)', display: 'block', mb: 0.5 }}>
                            💊 Dosage: {medicine.dosage || 'Not specified'}
                          </Typography>
                          <Typography component="span" variant="body2" sx={{ color: 'rgba(0, 0, 0, 0.8)', display: 'block', mb: 0.5 }}>
                            ⏰ Frequency: {medicine.frequency || 'Not specified'}
                          </Typography>
                          <Typography component="span" variant="body2" sx={{ color: 'rgba(0, 0, 0, 0.8)', display: 'block', mb: 0.5 }}>
                            📅 Duration: {medicine.duration || 'Not specified'}
                          </Typography>
                          {medicine.instructions && (
                            <Typography component="span" variant="body2" sx={{ color: 'rgba(0, 0, 0, 0.8)', display: 'block' }}>
                              📝 Instructions: {medicine.instructions}
                            </Typography>
                          )}
                        </Box>
                      }
                    />
                  </ListItem>
                  {index < response.medicines.length - 1 && <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.2)' }} />}
                </React.Fragment>
              ))}
            </List>
          </Paper>
        )}

        {/* Prescription Date */}
        {response.date && (
          <Paper sx={{ 
            padding: 3,
            background: 'rgba(255, 255, 255, 0.15)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: 2
          }}>
            <Typography variant="h6" fontWeight="bold" sx={{ color: 'black', mb: 1 }}>
              📅 Prescription Date
            </Typography>
            <Typography variant="body1" sx={{ color: 'rgba(0, 0, 0, 0.8)' }}>
              {response.date}
            </Typography>
          </Paper>
        )}
      </Box>
    );
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center', 
      padding: 2,
      minHeight: '100vh'
    }}>
      <Paper sx={{ 
        padding: 4, 
        textAlign: 'center', 
        width: '100%', 
        maxWidth: 900,
        borderRadius: 3,
        background: 'rgba(255, 255, 255, 0.25)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.18)',
        boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-5px)',
          boxShadow: '0 15px 40px 0 rgba(31, 38, 135, 0.5)',
        }
      }}>
        <Box sx={{ mb: 3 }}>
          <Typography 
            variant="h4" 
            gutterBottom 
            sx={{ 
              fontWeight: 700,
              background: 'linear-gradient(45deg, #667eea, #764ba2)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              mb: 2
            }}
          >
            ⚕ Medico Prescription Analyzer
          </Typography>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
            Upload your prescription image for AI-powered analysis
          </Typography>
        </Box>

        <div 
          {...getRootProps()} 
          className="upload-area"
          style={{ 
            border: '3px dashed rgba(255, 255, 255, 0.6)',
            borderRadius: 20,
            padding: '40px',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            background: 'rgba(255, 255, 255, 0.1)',
            backdropFilter: 'blur(10px)',
            marginBottom: 20
          }}
        >
          <input {...getInputProps()} />
          <Box sx={{ mb: 2 }}>
            <Typography variant="h2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mb: 1 }}>
              📄
            </Typography>
          </Box>
          <Typography variant="h6" sx={{ color: 'white', mb: 1, fontWeight: 600 }}>
            Drag & drop your prescription here
          </Typography>
          <Typography variant="body1" sx={{ color: 'rgba(255, 255, 255, 0.8)' }}>
            or click to browse files
          </Typography>
          <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.6)', mt: 1 }}>
            Supports: JPG, PNG, JPEG (Max 5MB)
          </Typography>
        </div>

        {file && (
          <Box sx={{ 
            marginTop: 3,
            padding: 2,
            borderRadius: 2,
            background: 'rgba(102, 126, 234, 0.1)',
            border: '1px solid rgba(102, 126, 234, 0.3)'
          }}>
            <Typography variant="body1" sx={{ color: 'white' }}>
              📎 Selected: <strong>{file.name}</strong>
            </Typography>
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.8)', mt: 0.5 }}>
              Size: {(file.size / 1024 / 1024).toFixed(2)} MB
            </Typography>
          </Box>
        )}

        <Button
          variant="contained"
          size="large"
          sx={{ 
            marginTop: 3,
            padding: '12px 32px',
            fontSize: '1.1rem',
            borderRadius: 3,
            background: 'linear-gradient(45deg, #667eea, #764ba2)',
            boxShadow: '0 4px 15px 0 rgba(102, 126, 234, 0.4)',
            '&:hover': {
              background: 'linear-gradient(45deg, #5a6fd8, #6a4190)',
              transform: 'translateY(-2px)',
              boxShadow: '0 6px 20px 0 rgba(102, 126, 234, 0.6)',
            },
            '&:disabled': {
              background: 'rgba(255, 255, 255, 0.3)',
              color: 'rgba(255, 255, 255, 0.6)',
            }
          }}
          onClick={handleUpload}
          disabled={loading || !file}
        >
          {loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CircularProgress size={20} color="inherit" />
              <span>Analyzing...</span>
            </Box>
          ) : (
            '🔍 Analyze Prescription'
          )}
        </Button>

        {errorMessage && (
          <Box sx={{ 
            marginTop: 3,
            padding: 2,
            borderRadius: 2,
            background: 'rgba(244, 67, 54, 0.1)',
            border: '1px solid rgba(244, 67, 54, 0.3)'
          }}>
            <Typography variant="body2" sx={{ color: '#ffcdd2' }}>
              ⚠️ {errorMessage}
            </Typography>
          </Box>
        )}

        {response && (
          <Box sx={{ 
            mt: 4,
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: 3,
            padding: 3,
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)'
          }}>
            <Tabs 
              value={activeTab} 
              onChange={(e, newValue) => setActiveTab(newValue)} 
              centered
              sx={{
                '& .MuiTab-root': {
                  color: 'rgba(255, 255, 255, 0.7)',
                  fontWeight: 600,
                  '&.Mui-selected': {
                    color: 'white',
                  }
                },
                '& .MuiTabs-indicator': {
                  backgroundColor: '#667eea',
                  height: 3,
                  borderRadius: 2
                }
              }}
            >
              <Tab label="📋 Prescription Details" />
              <Tab label="🧾 Generate Invoice" />
            </Tabs>
            
            {activeTab === 0 && renderPrescriptionData()}
            {activeTab === 1 && <InvoiceGenerator prescriptionData={response} />}
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default PrescriptionUpload;

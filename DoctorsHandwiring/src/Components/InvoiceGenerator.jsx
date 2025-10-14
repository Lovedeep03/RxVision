import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Alert
} from '@mui/material';
import { Add as AddIcon, Delete as DeleteIcon, Download as DownloadIcon, Edit as EditIcon, Search as SearchIcon } from '@mui/icons-material';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const InvoiceGenerator = ({ prescriptionData }) => {
	// Lightweight quote list for default notes
	const defaultQuotes = [
		"Healing is a matter of time, but it is sometimes also a matter of opportunity.",
		"Take care of your body. It’s the only place you have to live.",
		"Small steps every day lead to big changes.",
		"Medicine cures diseases, but only doctors can cure patients.",
		"Your health is an investment, not an expense."
	];

	// Estimate a reasonable price using name/description content (never return 0)
	const estimatePriceByContent = (name = '', description = '') => {
		const lowerName = name.toLowerCase();
		const lowerDesc = description.toLowerCase();

		// Strong keyword-based baselines
		if (lowerDesc.includes('inhaler') || lowerName.includes('inhaler')) return 250;
		if (lowerDesc.includes('syrup') || lowerName.includes('syrup')) return 120;
		if (lowerDesc.includes('injection') || lowerName.includes('injection')) return 180;
		if (lowerDesc.includes('capsule') || lowerName.includes('capsule')) return 90;
		if (lowerDesc.includes('tablet') || lowerName.includes('tablet')) return 80;

		// Antibiotic/common brand heuristics
		if (/(amoxicillin|azithromycin|cef|cipro|doxy)/.test(lowerName)) return 110;
		if (/(paracetamol|acetaminophen|calpol)/.test(lowerName)) return 25;
		if (/(omeprazole|pantoprazole)/.test(lowerName)) return 40;

		// Dosage-based heuristic: e.g., "500mg", "250 mg", "5 ml"
		const mgMatch = lowerDesc.match(/(\d{2,4})\s*mg/);
		if (mgMatch) {
			const mg = parseInt(mgMatch[1], 10);
			// Base price roughly proportional to strength
			const est = Math.max(20, Math.round((mg * 0.15) / 5) * 5);
			return est;
		}
		const mlMatch = lowerDesc.match(/(\d{2,4})\s*ml/);
		if (mlMatch) {
			const ml = parseInt(mlMatch[1], 10);
			const est = Math.max(30, Math.round((ml * 0.5) / 5) * 5);
			return est;
		}

		// Fallback generic price
		return 75;
	};
	const [invoiceData, setInvoiceData] = useState({
    invoiceNumber: `INV-${Date.now()}`,
    date: new Date().toISOString().split('T')[0],
    dueDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    patientName: prescriptionData?.patient?.name || '',
    patientAge: prescriptionData?.patient?.age || '',
    doctorName: prescriptionData?.doctor?.name || '',
    doctorSpecialization: prescriptionData?.doctor?.specialization || '',
		items: prescriptionData?.medicines?.map((medicine, index) => ({
      id: index + 1,
      name: medicine.name || 'Medicine',
			description: `${medicine.dosage || ''} - ${medicine.frequency || ''} - ${medicine.duration || ''}`,
      quantity: 1,
			price: 0, // placeholder; will be estimated below in useEffect
			total: 0,
      isSearching: false,
      searchError: null
    })) || [],
    subtotal: 0,
    tax: 0,
    total: 0,
		notes: defaultQuotes[Math.floor(Math.random() * defaultQuotes.length)]
  });

  const [editingItem, setEditingItem] = useState(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [searchingAll, setSearchingAll] = useState(false);
  const invoiceRef = useRef();

  // Medicine price database (simplified - in real app, this would come from API)
  const medicinePrices = {
    'paracetamol': { price: 15, description: '500mg tablet' },
    'calpol': { price: 25, description: '500mg syrup' },
    'levolin': { price: 180, description: 'Inhaler' },
    'meftal': { price: 45, description: '500mg tablet' },
    'amoxicillin': { price: 120, description: '500mg capsule' },
    'azithromycin': { price: 85, description: '500mg tablet' },
    'omeprazole': { price: 35, description: '20mg capsule' },
    'metformin': { price: 55, description: '500mg tablet' },
    'atenolol': { price: 25, description: '50mg tablet' },
    'amlodipine': { price: 30, description: '5mg tablet' },
    'losartan': { price: 40, description: '50mg tablet' },
    'simvastatin': { price: 65, description: '20mg tablet' },
    'aspirin': { price: 20, description: '75mg tablet' },
    'vitamin d': { price: 150, description: '1000 IU tablet' },
    'calcium': { price: 80, description: '500mg tablet' },
    'iron': { price: 45, description: '100mg tablet' },
    'folic acid': { price: 25, description: '5mg tablet' },
    'b complex': { price: 35, description: 'Tablet' },
    'vitamin c': { price: 30, description: '500mg tablet' },
    'zinc': { price: 40, description: '20mg tablet' }
  };

	const searchMedicinePrice = async (medicineName, index) => {
    if (!medicineName) return;

    const newItems = [...invoiceData.items];
    newItems[index] = { ...newItems[index], isSearching: true, searchError: null };
    setInvoiceData({ ...invoiceData, items: newItems });

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      const searchTerm = medicineName.toLowerCase().trim();
      let found = false;

      // Search in our database
      for (const [key, value] of Object.entries(medicinePrices)) {
        if (searchTerm.includes(key) || key.includes(searchTerm)) {
          newItems[index] = {
            ...newItems[index],
            price: value.price,
            description: value.description,
            isSearching: false,
            searchError: null
          };
          found = true;
          break;
        }
      }

			if (!found) {
				const estimated = estimatePriceByContent(newItems[index].name, newItems[index].description);
				newItems[index] = {
					...newItems[index],
					price: estimated,
					description: newItems[index].description || 'Standard dosage',
					isSearching: false,
					searchError: null
				};
			}

      // Recalculate totals
      const totals = calculateTotals(newItems);
      setInvoiceData({ ...invoiceData, items: newItems, ...totals });

    } catch (error) {
      newItems[index] = {
        ...newItems[index],
        isSearching: false,
        searchError: 'Failed to fetch price'
      };
      setInvoiceData({ ...invoiceData, items: newItems });
    }
  };

  const searchAllMedicines = async () => {
    setSearchingAll(true);
    
    for (let i = 0; i < invoiceData.items.length; i++) {
      if (invoiceData.items[i].name && !invoiceData.items[i].price) {
        await searchMedicinePrice(invoiceData.items[i].name, i);
        // Small delay between searches
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }
    
    setSearchingAll(false);
  };

	const calculateTotals = (items) => {
    const subtotal = items.reduce((sum, item) => sum + (item.quantity * item.price), 0);
    const tax = subtotal * 0.05; // 5% tax
    const total = subtotal + tax;
    return { subtotal, tax, total };
  };

	// Initialize estimated prices and totals on mount so no item stays at 0
	useEffect(() => {
		if (invoiceData.items.length > 0) {
			const initialized = invoiceData.items.map(it => {
				const price = it.price && it.price > 0 ? it.price : estimatePriceByContent(it.name, it.description);
				return { ...it, price, total: price * (it.quantity || 1) };
			});
			const totals = calculateTotals(initialized);
			setInvoiceData(prev => ({ ...prev, items: initialized, ...totals }));
		}
	// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

  const handleItemChange = (index, field, value) => {
    const newItems = [...invoiceData.items];
    newItems[index] = { ...newItems[index], [field]: value };
    
    if (field === 'quantity' || field === 'price') {
      newItems[index].total = newItems[index].quantity * newItems[index].price;
    }
    
    const totals = calculateTotals(newItems);
    setInvoiceData({
      ...invoiceData,
      items: newItems,
      ...totals
    });
  };

  const addItem = () => {
    const newItem = {
      id: invoiceData.items.length + 1,
      name: '',
      description: '',
      quantity: 1,
      price: 0,
      total: 0,
      isSearching: false,
      searchError: null
    };
    
    const newItems = [...invoiceData.items, newItem];
    const totals = calculateTotals(newItems);
    
    setInvoiceData({
      ...invoiceData,
      items: newItems,
      ...totals
    });
  };

  const removeItem = (index) => {
    const newItems = invoiceData.items.filter((_, i) => i !== index);
    const totals = calculateTotals(newItems);
    
    setInvoiceData({
      ...invoiceData,
      items: newItems,
      ...totals
    });
  };

  const editItem = (index) => {
    setEditingItem({ ...invoiceData.items[index], index });
    setEditDialogOpen(true);
  };

  const saveEditedItem = () => {
    if (editingItem) {
      const newItems = [...invoiceData.items];
      newItems[editingItem.index] = {
        ...editingItem,
        total: editingItem.quantity * editingItem.price
      };
      
      const totals = calculateTotals(newItems);
      setInvoiceData({
        ...invoiceData,
        items: newItems,
        ...totals
      });
    }
    setEditDialogOpen(false);
    setEditingItem(null);
  };

  const generatePDF = async () => {
    if (!invoiceRef.current) return;

    try {
      // Temporarily hide action buttons and action column for PDF generation
      const actionButtons = invoiceRef.current.querySelectorAll('.action-buttons');
      const actionHeaders = invoiceRef.current.querySelectorAll('.action-header');
      const originalDisplay = [];
      const originalHeaderDisplay = [];
      
      // Hide action buttons
      actionButtons.forEach((button, index) => {
        originalDisplay[index] = button.style.display;
        button.style.display = 'none';
      });

      // Hide action column headers
      actionHeaders.forEach((header, index) => {
        originalHeaderDisplay[index] = header.style.display;
        header.style.display = 'none';
      });

      // Add comprehensive print styles temporarily
      const style = document.createElement('style');
      style.textContent = `
        @media print {
          .invoice-container { 
            page-break-inside: avoid; 
            break-inside: avoid; 
            max-height: none !important;
            overflow: visible !important;
          }
          .totals-section { 
            page-break-inside: avoid; 
            break-inside: avoid; 
            page-break-before: auto;
            page-break-after: auto;
            display: flex !important;
            flex-direction: row !important;
            position: relative !important;
          }
          .totals-box {
            page-break-inside: avoid; 
            break-inside: avoid;
            page-break-before: auto;
            page-break-after: auto;
            display: block !important;
            position: relative !important;
          }
          .invoice-table { 
            page-break-inside: avoid; 
            break-inside: avoid; 
            max-height: none !important;
            overflow: visible !important;
          }
          .invoice-item { 
            page-break-inside: avoid; 
            break-inside: avoid; 
          }
          .action-buttons {
            display: none !important;
          }
          .action-header {
            display: none !important;
          }
          table {
            width: 100% !important;
            table-layout: auto !important;
          }
          td, th {
            padding: 8px !important;
            word-wrap: break-word !important;
            white-space: normal !important;
          }
        }
        
        .force-page-break {
          page-break-before: always;
          break-before: page;
        }
        
        .keep-together {
          page-break-inside: avoid;
          break-inside: avoid;
        }
        
        /* Force totals to stay together */
        .totals-section {
          page-break-inside: avoid !important;
          break-inside: avoid !important;
        }
        
        .totals-box {
          page-break-inside: avoid !important;
          break-inside: avoid !important;
        }
      `;
      document.head.appendChild(style);

      // Force a small delay to ensure styles are applied
      await new Promise(resolve => setTimeout(resolve, 100));

      const canvas = await html2canvas(invoiceRef.current, {
        scale: 1.5,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#ffffff',
        logging: false,
        removeContainer: true,
        width: invoiceRef.current.scrollWidth,
        height: invoiceRef.current.scrollHeight, // Use full height to capture all content
        scrollX: 0,
        scrollY: 0
      });

      // Remove temporary styles
      document.head.removeChild(style);

      // Restore action buttons
      actionButtons.forEach((button, index) => {
        button.style.display = originalDisplay[index];
      });

      // Restore action column headers
      actionHeaders.forEach((header, index) => {
        header.style.display = originalHeaderDisplay[index];
      });

      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      const imgWidth = 210;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      const pageHeight = 295; // A4 height in mm

      // If content fits on one page
      if (imgHeight <= pageHeight) {
        pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);
      } else {
        // Split content across multiple pages
        let heightLeft = imgHeight;
        let position = 0;
        let page = 1;

        while (heightLeft >= pageHeight) {
          pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
          heightLeft -= pageHeight;
          position -= pageHeight;
          
          if (heightLeft >= pageHeight) {
            pdf.addPage();
            page++;
          }
        }

        // Add remaining content
        if (heightLeft > 0) {
          pdf.addPage();
          pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
        }
      }

      pdf.save(`Invoice-${invoiceData.invoiceNumber}.pdf`);
    } catch (error) {
      console.error('Error generating PDF:', error);
      alert('Error generating PDF. Please try again.');
    }
  };

  // Auto-search medicines when component mounts
  useEffect(() => {
    if (invoiceData.items.length > 0) {
      setTimeout(() => {
        searchAllMedicines();
      }, 1000);
    }
  }, []);

  return (
    <Box sx={{ p: 3 }}>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom align="center">
          Medico Pharma Invoice Generator
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Invoice Number"
              value={invoiceData.invoiceNumber}
              onChange={(e) => setInvoiceData({ ...invoiceData, invoiceNumber: e.target.value })}
              margin="normal"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Date"
              type="date"
              value={invoiceData.date}
              onChange={(e) => setInvoiceData({ ...invoiceData, date: e.target.value })}
              margin="normal"
              InputLabelProps={{ shrink: true }}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Patient Name"
              value={invoiceData.patientName}
              onChange={(e) => setInvoiceData({ ...invoiceData, patientName: e.target.value })}
              margin="normal"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Patient Age"
              value={invoiceData.patientAge}
              onChange={(e) => setInvoiceData({ ...invoiceData, patientAge: e.target.value })}
              margin="normal"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Doctor Name"
              value={invoiceData.doctorName}
              onChange={(e) => setInvoiceData({ ...invoiceData, doctorName: e.target.value })}
              margin="normal"
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Doctor Specialization"
              value={invoiceData.doctorSpecialization}
              onChange={(e) => setInvoiceData({ ...invoiceData, doctorSpecialization: e.target.value })}
              margin="normal"
            />
          </Grid>
        </Grid>
      </Paper>

             {/* Invoice Preview */}
       <Paper ref={invoiceRef} className="invoice-container" sx={{ 
         p: 3, 
         mb: 3, 
         backgroundColor: 'white', 
         position: 'relative', 
         overflow: 'visible',
         minHeight: 'fit-content',
         '@media print': {
           pageBreakInside: 'avoid',
           breakInside: 'avoid',
           pageBreakBefore: 'auto',
           pageBreakAfter: 'auto',
           maxHeight: 'none',
           overflow: 'visible'
         }
       }}>
        {/* Medical Logo Background */}
                 <Box sx={{
           position: 'absolute',
           top: '50%',
           left: '50%',
           transform: 'translate(-50%, -50%)',
           opacity: 0.08,
           zIndex: 0,
           pointerEvents: 'none'
         }}>
           <Typography variant="h1" sx={{ fontSize: '400px', color: '#1976d2', fontWeight: 'bold' }}>
             ⚕
           </Typography>
         </Box>

                 <Box sx={{ position: 'relative', zIndex: 1 }}>
           <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
             <Box sx={{ flex: 1 }}>
               <Typography variant="h4" color="primary" gutterBottom sx={{ fontWeight: 'bold', textAlign: 'left', mb: 1 }}>
                 MEDICO PHARMA
               </Typography>
               <Typography variant="body2" sx={{ mb: 0.25 }}>Invoice #: {invoiceData.invoiceNumber}</Typography>
               <Typography variant="body2" sx={{ mb: 0.25 }}>Date: {invoiceData.date}</Typography>
              {/* Due Date removed as requested */}
             </Box>
             <Box sx={{ textAlign: 'right', flex: 1 }}>
               <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', mb: 1 }}>From:</Typography>
               <Typography variant="body2" sx={{ mb: 0.25 }}>Dr. {invoiceData.doctorName}</Typography>
               <Typography variant="body2" sx={{ mb: 0.25 }}>{invoiceData.doctorSpecialization}</Typography>
               <Typography variant="body2" sx={{ mb: 0.25 }}>Medical Professional</Typography>
             </Box>
           </Box>

                     <Divider sx={{ my: 2, borderWidth: 2 }} />

           <Box sx={{ mb: 2 }}>
             <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold', textAlign: 'left', mb: 1 }}>Bill To:</Typography>
             <Typography variant="body2" sx={{ mb: 0.25, textAlign: 'left' }}>Patient: {invoiceData.patientName}</Typography>
             <Typography variant="body2" sx={{ mb: 0.25, textAlign: 'left' }}>Age: {invoiceData.patientAge}</Typography>
           </Box>

                     <TableContainer className="invoice-table" sx={{ 
             mb: 2,
             maxHeight: 'auto',
             minHeight: 'fit-content',
             overflow: 'visible',
             pageBreakInside: 'avoid',
             breakInside: 'avoid',
             '@media print': {
               pageBreakInside: 'avoid',
               breakInside: 'avoid',
               maxHeight: 'none',
               overflow: 'visible'
             }
           }}>
            <Table>
              <TableHead>
                <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                  <TableCell sx={{ fontWeight: 'bold', borderBottom: '2px solid #1976d2' }}>Item</TableCell>
                  <TableCell sx={{ fontWeight: 'bold', borderBottom: '2px solid #1976d2' }}>Description</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 'bold', borderBottom: '2px solid #1976d2' }}>Quantity</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 'bold', borderBottom: '2px solid #1976d2' }}>Price (₹)</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 'bold', borderBottom: '2px solid #1976d2' }}>Total (₹)</TableCell>
                  <TableCell align="center" className="action-header" sx={{ fontWeight: 'bold', borderBottom: '2px solid #1976d2' }}>Actions</TableCell>
                </TableRow>
              </TableHead>
                          <TableBody>
              {invoiceData.items.map((item, index) => (
                <TableRow key={item.id} className="invoice-item">
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <TextField
                          size="small"
                          value={item.name}
                          onChange={(e) => handleItemChange(index, 'name', e.target.value)}
                          fullWidth
                        />
                        {item.isSearching && <CircularProgress size={16} />}
                        {item.searchError && (
                          <Alert severity="error" sx={{ fontSize: '0.7rem', p: 0.5 }}>
                            {item.searchError}
                          </Alert>
                        )}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <TextField
                        size="small"
                        value={item.description}
                        onChange={(e) => handleItemChange(index, 'description', e.target.value)}
                        fullWidth
                      />
                    </TableCell>
                    <TableCell align="right">
                      <TextField
                        size="small"
                        type="number"
                        value={item.quantity}
                        onChange={(e) => handleItemChange(index, 'quantity', parseFloat(e.target.value) || 0)}
                        sx={{ width: 80 }}
                      />
                    </TableCell>
                    <TableCell align="right">
                      <TextField
                        size="small"
                        type="number"
                        value={item.price}
                        onChange={(e) => handleItemChange(index, 'price', parseFloat(e.target.value) || 0)}
                        sx={{ width: 100 }}
                      />
                    </TableCell>
                    <TableCell align="right">
                      ₹{(item.quantity * item.price).toFixed(2)}
                    </TableCell>
                    <TableCell align="center" className="action-buttons">
                      <IconButton 
                        size="small" 
                        onClick={() => searchMedicinePrice(item.name, index)}
                        disabled={item.isSearching}
                      >
                        <SearchIcon />
                      </IconButton>
                      <IconButton size="small" onClick={() => editItem(index)}>
                        <EditIcon />
                      </IconButton>
                      <IconButton size="small" onClick={() => removeItem(index)} color="error">
                        <DeleteIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
                     </TableContainer>

           {/* Action Buttons */}
           <Box className="action-buttons" sx={{ mb: 2 }}>
             <Button
               variant="outlined"
               startIcon={<AddIcon />}
               onClick={addItem}
             >
               Add Item
             </Button>
             <Button
               variant="outlined"
               startIcon={searchingAll ? <CircularProgress size={16} /> : <SearchIcon />}
               onClick={searchAllMedicines}
               disabled={searchingAll}
               sx={{ ml: 2 }}
             >
               {searchingAll ? 'Searching...' : 'Search All Prices'}
             </Button>
           </Box>

                                {/* Totals Section - Kept together to prevent page breaks */}
                       <Box className="totals-section keep-together" sx={{ 
              display: 'flex', 
              justifyContent: 'flex-end', 
              mt: 2, 
              pageBreakInside: 'avoid',
              breakInside: 'avoid',
              position: 'relative',
              '@media print': {
                pageBreakInside: 'avoid',
                breakInside: 'avoid',
                pageBreakBefore: 'auto',
                pageBreakAfter: 'auto'
              }
            }}>
             <Box sx={{ 
               textAlign: 'right', 
               minWidth: '200px',
               pageBreakInside: 'avoid',
               breakInside: 'avoid',
               '@media print': {
                 pageBreakInside: 'avoid',
                 breakInside: 'avoid'
               }
             }}>
              <Box className="totals-box keep-together" sx={{ 
                border: '2px solid #1976d2', 
                borderRadius: 2, 
                p: 2, 
                backgroundColor: '#f8f9fa',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                pageBreakInside: 'avoid',
                breakInside: 'avoid',
                '@media print': {
                  pageBreakInside: 'avoid',
                  breakInside: 'avoid',
                  pageBreakBefore: 'auto',
                  pageBreakAfter: 'auto'
                }
              }}>
                <Typography variant="body1" sx={{ mb: 1, fontWeight: 'bold' }}>
                  Subtotal: ₹{invoiceData.subtotal.toFixed(2)}
                </Typography>
                <Typography variant="body1" sx={{ mb: 1, fontWeight: 'bold' }}>
                  Tax (5%): ₹{invoiceData.tax.toFixed(2)}
                </Typography>
                <Divider sx={{ my: 1 }} />
                <Typography variant="h5" color="primary" sx={{ fontWeight: 'bold' }}>
                  Total: ₹{invoiceData.total.toFixed(2)}
                </Typography>
              </Box>
            </Box>
          </Box>

                     <Box sx={{ mt: 2 }}>
             <TextField
               fullWidth
               multiline
               rows={2}
               label="Notes"
               value={invoiceData.notes}
               onChange={(e) => setInvoiceData({ ...invoiceData, notes: e.target.value })}
               sx={{ mb: 1 }}
             />
           </Box>
           
           {/* Minimal spacing */}
           <Box sx={{ height: '10px' }} />
        </Box>
      </Paper>

      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
        <Button
          variant="contained"
          color="primary"
          size="large"
          startIcon={<DownloadIcon />}
          onClick={generatePDF}
        >
          Download PDF Invoice
        </Button>
      </Box>

      {/* Edit Item Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Item</DialogTitle>
        <DialogContent>
          {editingItem && (
            <Box sx={{ pt: 1 }}>
              <TextField
                fullWidth
                label="Item Name"
                value={editingItem.name}
                onChange={(e) => setEditingItem({ ...editingItem, name: e.target.value })}
                margin="normal"
              />
              <TextField
                fullWidth
                label="Description"
                value={editingItem.description}
                onChange={(e) => setEditingItem({ ...editingItem, description: e.target.value })}
                margin="normal"
              />
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Quantity"
                    type="number"
                    value={editingItem.quantity}
                    onChange={(e) => setEditingItem({ ...editingItem, quantity: parseFloat(e.target.value) || 0 })}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    label="Price (₹)"
                    type="number"
                    value={editingItem.price}
                    onChange={(e) => setEditingItem({ ...editingItem, price: parseFloat(e.target.value) || 0 })}
                    margin="normal"
                  />
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button onClick={saveEditedItem} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default InvoiceGenerator;

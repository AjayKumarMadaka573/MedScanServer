const express = require('express');
const multer = require('multer');
const { execFile } = require('child_process');
const path = require('path');
require('dotenv').config();
const mongoose = require('mongoose');
const passport = require('passport');
const cors = require('cors');
const authRoutes = require('./routes/auth');
const mailRoutes = require('./routes/mail');

const app = express();
const port = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(passport.initialize());
require('./config/passport');

// Database connection
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('MongoDB connected'))
  .catch(err => console.log(err));


// Routes
app.use('/auth', authRoutes);
app.use('/mail',mailRoutes);



// Set up file upload
const storage = multer.diskStorage({
  destination: 'uploads/',
  filename: (_, file, cb) => cb(null, Date.now() + path.extname(file.originalname))
});

const upload = multer({ storage });


// Handle POST request for image prediction
app.post('/predict', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }
    
    console.log('Uploaded file:', req.file); // Log the file information for debugging
    const imgPath = req.file.path;
    const anacondaPython = 'C:\\Users\\AJAY KUMAR MADAKA\\anaconda3\\python.exe';

    // Execute Python script using execFile
    execFile(anacondaPython , ['predict.py', imgPath], (err, stdout, stderr) => {
        if (err) {
            console.error('Error executing Python script:', err);
            return res.status(500).json({ error: 'Error executing Python script', message: stderr });
        }
        
        // Parse the output (stdout) from Python
        try {
            const output = JSON.parse(stdout);
            console.log("Output"+output);
            res.json({ results: output });
        } catch (err) {
            console.error('Error parsing Python script output:', err);
            res.status(500).json({ error: 'Failed to parse output' });
        }
        
    });
});

// Start the server
app.listen(port, () => console.log(`Server running at http://localhost:${port}`));

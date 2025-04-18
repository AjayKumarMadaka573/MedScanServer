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
app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        // 1. Validate input
        if (!req.file) {
            console.error('No file uploaded');
            return res.status(400).json({ 
                status: 'error',
                error: 'No file uploaded',
                timestamp: new Date().toISOString()
            });
        }

        console.log('Uploaded file:', {
            originalname: req.file.originalname,
            mimetype: req.file.mimetype,
            size: req.file.size,
            path: req.file.path
        });

        const imgPath = req.file.path;
        const pythonCommand = process.env.PYTHON_COMMAND || 'python3'; // Configurable Python command

        // 2. Execute Python script with timeout
        const { stdout, stderr } = await new Promise((resolve, reject) => {
            execFile(
                pythonCommand,
                ['predict.py', imgPath],
                { timeout: 30000 }, // 30-second timeout
                (error, stdout, stderr) => {
                    if (error) {
                        console.error('Python execution error:', {
                            error: error.message,
                            stderr: stderr,
                            stdout: stdout
                        });
                        reject(new Error(`Python script failed: ${error.message}`));
                        return;
                    }
                    resolve({ stdout, stderr });
                }
            );
        });

        // 3. Clean up the uploaded file after processing
        try {
            await fs.promises.unlink(imgPath);
            console.log('Temporary file deleted:', imgPath);
        } catch (cleanupError) {
            console.error('File cleanup error:', cleanupError);
        }

        // 4. Parse and validate output
        try {
            const output = JSON.parse(stdout.trim());
            
            if (!output.status) {
                throw new Error('Invalid output format from Python script');
            }

            console.log('Prediction results:', {
                status: output.status,
                task: output.result?.task,
                prediction: output.result?.label
            });

            return res.json({
                status: 'success',
                data: output,
                metadata: {
                    processing_time: new Date(),
                    python_version: output.python_version || 'unknown'
                }
            });

        } catch (parseError) {
            console.error('Output parsing failed:', {
                error: parseError.message,
                rawOutput: stdout,
                stderr: stderr
            });
            
            return res.status(500).json({
                status: 'error',
                error: 'Output processing failed',
                details: {
                    message: parseError.message,
                    rawOutput: stdout.length > 200 ? stdout.substring(0, 200) + '...' : stdout
                }
            });
        }

    } catch (error) {
        console.error('Prediction endpoint error:', {
            error: error.message,
            stack: error.stack
        });

        return res.status(500).json({
            status: 'error',
            error: 'Prediction processing failed',
            message: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// Start the server
app.listen(port, () => console.log(`Server running at http://localhost:${port}`));

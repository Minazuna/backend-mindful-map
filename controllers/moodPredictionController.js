const { spawn } = require('child_process');
const path = require('path');
const MoodLog = require('../models/MoodLog');

exports.predictMood = async (req, res) => {
    try {
        console.log("User ID from request:", req.user._id); 

        const moodLogs = await MoodLog.find({ 
            user: req.user._id,
            date: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) }
        }).select('mood date activities -_id');

        console.log("Retrieved mood logs:", moodLogs);

        if (moodLogs.length < 7) {
            return res.status(200).json({
                success: true,
                predictions: {},
                message: 'Need at least one week of mood data for predictions'
            });
        }

        const formattedLogs = moodLogs.map(log => ({
            mood: log.mood.toLowerCase(),
            timestamp: log.date.toISOString(),
            activities: Array.isArray(log.activities) ? log.activities : []
        }));

        console.log("Formatted logs for Python:", formattedLogs);

        return new Promise((resolve, reject) => {
            // const pythonProcess = spawn(
            //     path.join(process.cwd(), '..', 'venv', 'Scripts', 'python'),
            //     [path.join(__dirname, '..', 'mood_prediction.py')],
            //     { stdio: ['pipe', 'pipe', 'pipe'] }
            // );

            // const pythonExecutable = process.platform === 'win32' 
            //     ? path.join(process.cwd(), '..', 'venv', 'Scripts', 'python.exe') 
            //     : path.join(process.cwd(), '..', 'venv', 'bin', 'python');

            // const pythonProcess = spawn(pythonExecutable, 
            //     [path.join(__dirname, '..', 'mood_prediction.py')],
            //     { stdio: ['pipe', 'pipe', 'pipe'] }
            // );

            const scriptPath = path.resolve(__dirname, '..', 'mood_prediction.py');
            const pythonProcess = spawn(pythonExecutable, [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] });

            let pythonData = '';
            let pythonError = '';

            pythonProcess.stdin.write(JSON.stringify(formattedLogs));
            pythonProcess.stdin.end();

            pythonProcess.stdout.on('data', (data) => {
                pythonData += data.toString();
                console.log("Python output received:", data.toString());
            });

            pythonProcess.stderr.on('data', (data) => {
                pythonError += data.toString();
                console.error("Python error:", data.toString());
            });

            pythonProcess.on('close', (code) => {
                console.log("Python process closed with code:", code);
                console.log("Final Python output:", pythonData);

                if (code !== 0) {
                    console.error('Python process error:', pythonError);
                    return res.status(500).json({
                        success: false,
                        message: 'Error generating mood predictions',
                        error: pythonError
                    });
                }

                try {
                    // Clean the output string
                    const cleanedData = pythonData.trim();
                    if (!cleanedData) {
                        throw new Error('No output from Python script');
                    }

                    const predictions = JSON.parse(cleanedData);

                    if (predictions.error) {
                        return res.status(500).json({
                            success: false,
                            message: predictions.error
                        });
                    }

                    res.json({
                        success: true,
                        predictions: predictions.daily_predictions,
                        insights: predictions.insights
                    });
                } catch (error) {
                    console.error('JSON Parse Error:', error);
                    console.error('Raw Python output:', pythonData);
                    res.status(500).json({
                        success: false,
                        message: 'Error processing prediction results',
                        error: error.message,
                        rawOutput: pythonData
                    });
                }
            });

            pythonProcess.on('error', (error) => {
                console.error('Failed to start Python process:', error);
                res.status(500).json({
                    success: false,
                    message: 'Failed to start prediction process',
                    error: error.message
                });
            });
        });

    } catch (error) {
        console.error('Controller Error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error while generating predictions',
            error: error.message
        });
    }
};
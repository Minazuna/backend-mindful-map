const fetch = require('node-fetch');
const MoodLog = require('../models/MoodLog');

exports.predictMood = async (req, res) => {
    try {
        console.log("User ID from request:", req.user._id); 

        // Get the user's mood logs from the database
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

        // Format logs for the Python service
        const formattedLogs = moodLogs.map(log => ({
            mood: log.mood.toLowerCase(),
            timestamp: log.date.toISOString(),
            activities: Array.isArray(log.activities) ? log.activities : []
        }));

        console.log("Formatted logs for Python service:", formattedLogs);

        // Forward the request to the Python service
        const pythonApiUrl = process.env.PYTHON_API_URL || 'https://mindful-map-backend-python.onrender.com';
        const token = req.headers.authorization;
        
        // Instead of making a direct API call to the Python service, 
        // we'll just forward the user's token so the Python service can fetch the logs itself
        const pythonResponse = await fetch(`${pythonApiUrl}/api/predict-mood`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': token
            }
        });

        const pythonData = await pythonResponse.json();
        console.log("Python service response:", pythonData);

        if (!pythonData.success) {
            return res.status(500).json({
                success: false,
                message: 'Error generating mood predictions',
                error: pythonData.message
            });
        }

        res.json({
            success: true,
            predictions: pythonData.predictions,
            insights: pythonData.insights
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

// Add a new endpoint to provide mood logs to the Python service
exports.getMoodLogs = async (req, res) => {
    try {
        const moodLogs = await MoodLog.find({ 
            user: req.user._id,
            date: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) }
        }).select('mood date activities -_id');

        return res.status(200).json({
            success: true,
            logs: moodLogs.map(log => ({
                mood: log.mood,
                date: log.date.toISOString(),
                activities: Array.isArray(log.activities) ? log.activities : []
            }))
        });
    } catch (error) {
        console.error('Error fetching mood logs:', error);
        return res.status(500).json({
            success: false,
            message: 'Error fetching mood logs',
            error: error.message
        });
    }
};
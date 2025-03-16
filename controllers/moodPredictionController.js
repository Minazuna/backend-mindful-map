const axios = require('axios');
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

        console.log("Formatted logs for Python service:", formattedLogs);

        // Get the Python service URL from environment variables
        const PYTHON_SERVICE_URL = `${process.env.VITE_PYTHON_API}/api/predict-mood`;
        console.log("Sending POST request to:", PYTHON_SERVICE_URL);

        // Send the data directly to the Python service
        const pythonResponse = await axios.post(PYTHON_SERVICE_URL, formattedLogs, {
            headers: {
                'Content-Type': 'application/json'
            },
            timeout: 60000 // 60 seconds timeout for ML processing
        });
        
        console.log("Python service response:", pythonResponse.data);
        
        if (!pythonResponse.data.success) {
            return res.status(500).json({
                success: false,
                message: pythonResponse.data.message || 'Error from prediction service'
            });
        }
        
        return res.json({
            success: true,
            predictions: pythonResponse.data.predictions || {},
            insights: pythonResponse.data.insights || {}
        });

    } catch (error) {
        console.error('Controller Error:', error);
        // Enhanced error logging
        if (error.response) {
            console.error('Response data:', error.response.data);
            console.error('Response status:', error.response.status);
        } else if (error.request) {
            console.error('No response received:', error.request);
        }
        res.status(500).json({
            success: false,
            message: 'Server error while generating predictions',
            error: error.message
        });
    }
};
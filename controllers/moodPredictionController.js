const axios = require('axios');
const MoodLog = require('../models/MoodLog');

// Use the exact URL from your environment variable
const PYTHON_SERVICE_URL = process.env.VITE_PYTHON_API;

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

        // Debug the URL we're using
        console.log("Making POST request to:", PYTHON_SERVICE_URL);

        // Send the data to the Python service
        const pythonResponse = await axios.post(PYTHON_SERVICE_URL, formattedLogs, {
            headers: {
                'Content-Type': 'application/json'
            },
            timeout: 30000 // 30 seconds timeout
        });
        
        console.log("Python service response:", pythonResponse.data);
        
        // Check the structure of the actual response
        if (!pythonResponse.data || (!pythonResponse.data.predictions && !pythonResponse.data.daily_predictions)) {
            console.error("Unexpected response format:", pythonResponse.data);
            return res.status(500).json({
                success: false,
                message: 'Invalid response format from prediction service'
            });
        }
        
        // Use whichever format the Python service returns
        const predictions = pythonResponse.data.daily_predictions || pythonResponse.data.predictions;
        const insights = pythonResponse.data.insights || {};
        
        return res.json({
            success: true,
            predictions: predictions,
            insights: insights
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
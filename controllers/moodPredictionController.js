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

        // Format the logs for the Flask API
        const formattedLogs = moodLogs.map(log => ({
            mood: log.mood.toLowerCase(),
            timestamp: log.date.toISOString(),
            activities: Array.isArray(log.activities) ? log.activities : []
        }));

        console.log("Formatted logs for Python API:", formattedLogs);

        // Call the Flask API
        const pythonApiUrl = process.env.PYTHON_API_URL || 'https://mindful-map-backend-python.onrender.com';
        const response = await axios.get(`${pythonApiUrl}/api/predict-mood`, formattedLogs, {
            headers: {
                'Content-Type': 'application/json',
                // Forward auth token if needed
                ...(req.headers.authorization && { 'Authorization': req.headers.authorization })
            }
        });

        // Return the predictions from the Flask API
        res.json({
            success: true,
            predictions: response.data.predictions,
            insights: response.data.insights
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
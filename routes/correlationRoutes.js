const express = require('express');
const { getWeeklyCorrelationForCorrelation, getWeeklyCorrelationForStatistics, getRecommendations, getWeeklyComparison } = require('../controllers/correlationController');
const { authMiddleware } = require('../middleware/authMiddleware'); 

const router = express.Router();

router.get('/weekly-correlation', authMiddleware, getWeeklyCorrelationForCorrelation);
router.get('/weekly-correlation-statistics', authMiddleware, getWeeklyCorrelationForStatistics);
// router.get('/monthly-correlation', authMiddleware, getMonthlyCorrelation);
router.post('/recommendations', authMiddleware, getRecommendations);
router.get('/weekly-comparison', authMiddleware, getWeeklyComparison);

module.exports = router;
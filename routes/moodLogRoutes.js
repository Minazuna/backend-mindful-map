const express = require('express');
const { authMiddleware, userMiddleware } = require('../middleware/authMiddleware');
const { saveMood, getAllMoodLogs, getPaginatedMoodLogs, checkMoodLogs } = require('../controllers/moodLogController');

const router = express.Router();

router.post('/mood-log', authMiddleware, userMiddleware, saveMood);
router.get('/mood-log', authMiddleware, userMiddleware, getAllMoodLogs);
router.get('/mood-log/paginated', authMiddleware, userMiddleware, getPaginatedMoodLogs);
router.get('/check-mood-logs', authMiddleware, userMiddleware, checkMoodLogs);

module.exports = router;
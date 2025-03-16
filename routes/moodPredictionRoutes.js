const express = require("express");
const { authMiddleware, userMiddleware } = require("../middleware/authMiddleware");
const { predictMood, getMoodLogs } = require("../controllers/moodPredictionController");

const router = express.Router();

router.get("/predict-mood", authMiddleware, userMiddleware, predictMood);
router.get("/mood-logs", authMiddleware, userMiddleware, getMoodLogs);

module.exports = router;
const express = require("express");
const { authMiddleware, userMiddleware } = require("../middleware/authMiddleware");
const { predictMood } = require("../controllers/moodPredictionController");

const router = express.Router();

router.get("/predict-mood", authMiddleware, userMiddleware, predictMood);

module.exports = router;
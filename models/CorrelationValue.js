const mongoose = require('mongoose');

const correlationvalueSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  correlationResults: [
    {
      correlationValue: Number,
      correlationMood: String,
      correlationActivity: String,
      correlationSocial: String,
      sleepQualityValue: Number,
      sleepQuality: String,
      healthStatus: String
    }
  ],
  recommendations: [String], // Added field for recommendations
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('CorrelationValue', correlationvalueSchema);
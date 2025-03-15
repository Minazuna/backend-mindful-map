const mongoose = require('mongoose');

const correlationSchema = new mongoose.Schema({
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
      sleepQualityValue: Number,
      sleepQuality: String
    }
  ],
  createdAt: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('Correlation', correlationSchema);
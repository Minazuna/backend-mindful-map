const mongoose = require('mongoose');

const JournalSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  images: { type: [String], default: [] },
  date: { type: Date, default: Date.now },
  time: { type: String, default: () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) },
  deleted: { type: Boolean, default: false },
  challenge: { type: Object, default: null },
  challengeData: { type: Object, default: null },
});

module.exports = mongoose.model('Journal', JournalSchema);
const mongoose = require('mongoose');

const promptSchema = new mongoose.Schema({
  question: { 
    type: String, 
    required: true, 
    unique: true 
},
  createdBy: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'User', required: true 
}, 
  createdAt: { 
    type: Date, 
    default: Date.now
},
  isUsed: { 
    type: Boolean, 
    default: false }
});

module.exports = mongoose.model('Prompt', promptSchema);
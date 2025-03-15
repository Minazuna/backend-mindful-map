const mongoose = require("mongoose");

const forumSchema = new mongoose.Schema({
  prompt: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Prompt",
    required: true,
  },
  discussions: [
    {
      user: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
      content: { type: String, required: true },
      createdAt: { type: Date, default: Date.now },
    },
  ],
  createdAt: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Forum", forumSchema);

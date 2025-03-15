const Prompt = require("../models/Prompt");
const Forum = require('../models/Forum');
const User = require('../models/User');
const moment = require("moment");
const mongoose = require('mongoose');
const { getFilter } = require('../helper.js');

exports.postComment = async (req, res) => {
  try {
    const { promptId, content } = req.body;
    const userId = req.user._id;

    if (!promptId || !content) {
      return res.status(400).json({ message: 'Prompt ID and content are required' });
    }

    if (!mongoose.Types.ObjectId.isValid(promptId)) {
      return res.status(400).json({ message: 'Invalid prompt ID format' });
    }

    // Check for profanity before saving
    const filter = getFilter();
    if (filter.check(content)) {
      return res.status(400).json({ 
        message: 'Your comment contains inappropriate language. Please remove any offensive words before posting.'
      });
    }

    const prompt = await Prompt.findById(promptId);
    if (!prompt) {
      return res.status(404).json({ message: 'Prompt not found' });
    }

    let forum = await Forum.findOne({ prompt: promptId });
    if (!forum) {
      forum = new Forum({ prompt: promptId, discussions: [] });
    }

    const newDiscussion = {
      user: userId,
      content: content, // Using original content since we've already verified it's clean
      createdAt: new Date(),
    };

    forum.discussions.push(newDiscussion);
    await forum.save();

    // Populate user data for the new discussion
    const populatedDiscussion = await Forum.findOne(
      { _id: forum._id, 'discussions._id': forum.discussions[forum.discussions.length - 1]._id }
    ).populate({
      path: 'discussions.user',
      select: 'name avatar',
    });

    const newDiscussionWithUser = populatedDiscussion.discussions[populatedDiscussion.discussions.length - 1];

    return res.status(201).json({
      message: 'Comment posted successfully',
      newDiscussion: newDiscussionWithUser,
    });
  } catch (error) {
    console.error('Error posting comment:', error);
    return res.status(500).json({ message: 'Error posting comment', error: error.message });
  }
};

exports.getTodaysPrompt = async (req, res) => {
  try {
    const today = moment().startOf("day");

    // Check if a prompt is already selected for today
    let todayPrompt = await Prompt.findOne({ createdAt: { $gte: today.toDate() } });

    if (!todayPrompt) {
      const unusedPrompts = await Prompt.find({ isUsed: false });

      if (unusedPrompts.length === 0) {
        return res.json({ message: "No more unused prompts available." });
      }

      const randomIndex = Math.floor(Math.random() * unusedPrompts.length);
      todayPrompt = unusedPrompts[randomIndex];

      todayPrompt.isUsed = true;
      todayPrompt.createdAt = new Date();
      await todayPrompt.save();
    }

    res.json(todayPrompt);
  } catch (error) {
    console.error("Error selecting today's prompt:", error);
    res.status(500).json({ error: "Failed to fetch today's prompt." });
  }
};

exports.getCurrentUser = async (req, res) => {
    try {
      const userId = req.user._id;
  
      if (!userId) {
        return res.status(401).json({
          message: 'User not authenticated'
        });
      }
  
      const user = await User.findById(userId)
        .select('name email avatar isDeactivated deactivatedAt'); 
  
      if (!user) {
        return res.status(404).json({
          message: 'User not found'
        });
      }
  
      res.status(200).json(user);
    } catch (error) {
      console.error('Error fetching user:', error);
      res.status(500).json({
        message: 'Error retrieving user data',
        error: error.message
      });
    }
  };

exports.getForumDiscussions = async (req, res) => {
  try {
    const { promptId } = req.params;

    if (!mongoose.Types.ObjectId.isValid(promptId)) {
      return res.status(400).json({
        message: 'Invalid prompt ID format'
      });
    }

    const forum = await Forum.findOne({ prompt: promptId })
      .populate({
        path: 'discussions.user',
        select: 'name avatar'
      });

    if (!forum) {
      return res.status(200).json({ discussions: [] });
    }

    return res.status(200).json({
      discussions: forum.discussions.sort((a, b) => b.createdAt - a.createdAt)
    });
  } catch (error) {
    console.error('Error getting forum discussions:', error);
    return res.status(500).json({
      message: 'Error retrieving forum discussions',
      error: error.message
    });
  }
};

// exports.postComment = async (req, res) => {
//   try {
//     const { promptId, content } = req.body;
//     const userId = req.user._id;

//     if (!promptId || !content) {
//       return res.status(400).json({
//         message: 'Prompt ID and content are required'
//       });
//     }

//     if (!mongoose.Types.ObjectId.isValid(promptId)) {
//       return res.status(400).json({
//         message: 'Invalid prompt ID format'
//       });
//     }

//     // Find the prompt to ensure it exists
//     const prompt = await Prompt.findById(promptId);
//     if (!prompt) {
//       return res.status(404).json({
//         message: 'Prompt not found'
//       });
//     }

//     // Find or create forum for this prompt
//     let forum = await Forum.findOne({ prompt: promptId });
    
//     if (!forum) {
//       forum = new Forum({
//         prompt: promptId,
//         discussions: []
//       });
//     }

//     const newDiscussion = {
//       user: userId,
//       content: content,
//       createdAt: new Date()
//     };

//     // Add to discussions array
//     forum.discussions.push(newDiscussion);
//     await forum.save();

//     // Populate user data for the new discussion
//     const populatedDiscussion = await Forum.findOne(
//       { 
//         _id: forum._id, 
//         'discussions._id': forum.discussions[forum.discussions.length - 1]._id 
//       }
//     ).populate({
//       path: 'discussions.user',
//       select: 'name avatar'
//     });

//     const newDiscussionWithUser = populatedDiscussion.discussions[populatedDiscussion.discussions.length - 1];

//     return res.status(201).json({
//       message: 'Comment posted successfully',
//       newDiscussion: newDiscussionWithUser
//     });
//   } catch (error) {
//     console.error('Error posting comment:', error);
//     return res.status(500).json({
//       message: 'Error posting comment',
//       error: error.message
//     });
//   }
// };

exports.deleteComment = async (req, res) => {
  try {
    const { promptId, commentId } = req.params;
    const userId = req.user._id;

    if (!mongoose.Types.ObjectId.isValid(promptId) || !mongoose.Types.ObjectId.isValid(commentId)) {
      return res.status(400).json({
        message: 'Invalid ID format'
      });
    }

    const forum = await Forum.findOne({ prompt: promptId });
    
    if (!forum) {
      return res.status(404).json({
        message: 'Forum not found'
      });
    }

    // Find the comment and ensure the user is the owner
    const commentIndex = forum.discussions.findIndex(
      discussion => 
        discussion._id.toString() === commentId && 
        discussion.user.toString() === userId.toString()
    );

    if (commentIndex === -1) {
      return res.status(403).json({
        message: 'Comment not found or you do not have permission to delete it'
      });
    }
    
    // Remove the comment
    forum.discussions.splice(commentIndex, 1);
    await forum.save();

    return res.status(200).json({
      message: 'Comment deleted successfully'
    });
  } catch (error) {
    console.error('Error deleting comment:', error);
    return res.status(500).json({
      message: 'Error deleting comment',
      error: error.message
    });
  }
};

exports.getPastForums = async (req, res) => {
  try {
    const userId = req.user._id;
    const today = moment().startOf("day");

    const forums = await Forum.find({
      'discussions.0': { $exists: true }, 
    })
    .populate({
      path: 'prompt',
      match: { createdAt: { $lt: today.toDate() } }, // Only get prompts from before today
      select: 'question createdAt'
    })
    .populate({
      path: 'discussions.user',
      select: 'name avatar'
    })
    .sort({ 'prompt.createdAt': -1 }); 

    const validForums = forums.filter(forum => forum.prompt !== null);

    return res.status(200).json(validForums);
  } catch (error) {
    console.error('Error getting past forums:', error);
    return res.status(500).json({
      message: 'Error retrieving past forums',
      error: error.message
    });
  }
};
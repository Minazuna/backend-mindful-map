const express = require("express");
const { authMiddleware, userMiddleware } = require('../middleware/authMiddleware');
const { getTodaysPrompt, getPastForums, getCurrentUser, getForumDiscussions, postComment, deleteComment } = require("../controllers/forumController");

const router = express.Router();

router.get('/user', authMiddleware, getCurrentUser);
router.get("/todays-prompt", authMiddleware, userMiddleware, getTodaysPrompt);
router.get("/past-forums", authMiddleware, userMiddleware, getPastForums);
router.get('/:promptId', authMiddleware, userMiddleware, getForumDiscussions);
router.post('/comment', authMiddleware, userMiddleware, postComment);
router.delete('/:promptId/comment/:commentId', authMiddleware, userMiddleware, deleteComment);

module.exports = router;

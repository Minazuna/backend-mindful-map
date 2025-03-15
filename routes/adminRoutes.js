const express = require('express');
const router = express.Router();
const { authMiddleware, adminMiddleware } = require('../middleware/authMiddleware');
const adminController = require('../controllers/adminController');

router.get('/dashboard', authMiddleware, adminMiddleware, adminController.dashboard);
router.get('/monthly-users', authMiddleware, adminMiddleware, adminController.getMonthlyUsers);
router.get('/active-users', authMiddleware, adminMiddleware, adminController.getActiveUsers);
router.get('/inactive-users', authMiddleware, adminMiddleware, adminController.getInactiveUsers); // Added route for inactive users
router.get('/users', authMiddleware, adminMiddleware, adminController.getUsers);
router.get('/user/:userId/moodlogs', authMiddleware, adminMiddleware, adminController.getUserMoodLogs);

// router.get('/statistics/', authMiddleware, adminMiddleware, adminController.getStatistics);

router.get('/daily-mood-logs', authMiddleware, adminMiddleware, adminController.getDailyMoodLogs);
router.get('/daily-journal-logs', authMiddleware, adminMiddleware, adminController.getDailyJournalLogs);
router.get('/daily-forum-engagement', adminController.getDailyForumEngagement);
router.get('/weekly-forum-engagement', adminController.getWeeklyForumEngagement);

router.post("/soft-delete", authMiddleware, adminMiddleware, adminController.softDelete);
router.get("/reactivate", authMiddleware, adminMiddleware, adminController.reactivate);
router.post("/reactivate", authMiddleware, adminMiddleware, adminController.reactivate);
router.post("/bulk-delete", authMiddleware, adminMiddleware, adminController.bulkDelete);
router.post('/check-expired-grace-periods', adminController.checkExpiredGracePeriods);

router.get("/prompts",  authMiddleware, adminMiddleware, adminController.getAllPrompts);
router.post("/add-prompt",  authMiddleware, adminMiddleware, adminController.addPrompt);
router.delete("/:id",  authMiddleware, adminMiddleware, adminController.deletePrompt);

// New route for fetching correlation values
router.get('/correlation-values', authMiddleware, adminMiddleware, adminController.getCorrelationValues);
router.get('/weekly-correlation-values', authMiddleware, adminMiddleware, adminController.getWeeklyCorrelationValues);
router.get('/weekly-forum-posts', authMiddleware, adminMiddleware, adminController.getWeeklyForumPosts);
router.get('/active-vs-inactive-users', authMiddleware, adminMiddleware, adminController.getActiveVsInactiveUsers);
module.exports = router;
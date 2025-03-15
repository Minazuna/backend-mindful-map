const express = require('express');
const { authMiddleware, userMiddleware } = require('../middleware/authMiddleware');
const { createJournalEntry, getJournalEntries, getJournalEntryById, updateJournalEntry, deleteJournalEntry, getJournalPrompt, upload } = require('../controllers/journalController');

const router = express.Router();

router.post('/journal', authMiddleware, userMiddleware, upload.array('images', 10), createJournalEntry);
router.get('/journals', authMiddleware, userMiddleware, getJournalEntries);
router.get('/journal/:id', authMiddleware, userMiddleware, getJournalEntryById);
router.put('/journal/:id', authMiddleware, userMiddleware, upload.array('images', 10), updateJournalEntry);
router.delete('/journal/:id', authMiddleware, userMiddleware, deleteJournalEntry);

module.exports = router;
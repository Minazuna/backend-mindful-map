const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');

// Run analysis script
router.get('/run-analysis', (req, res) => {
  const pythonProcess = spawn('python', ['predict_mood.py']);

  pythonProcess.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
    res.json({ message: 'Analysis completed' });
  });
});

module.exports = router;
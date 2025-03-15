const cron = require("node-cron");
const { processExpiredGracePeriods } = require("./accountService");

// Run every hour to check and process expired grace periods
const initScheduledTasks = () => {
  cron.schedule('0 * * * *', async () => {
    try {
      console.log('Running scheduled check for expired grace periods...');
      const result = await processExpiredGracePeriods();
      console.log(`Processed ${result.processedCount} expired grace periods`);
    } catch (error) {
      console.error("Error in scheduled check for expired grace periods:", error);
    }
  });

  console.log("Scheduled tasks initialized");
};

module.exports = { initScheduledTasks };
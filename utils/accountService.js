const User = require("../models/User");
const sendMail = require("../utils/mailer");
const { 
  deactivationNoticeTemplate, 
  accountDisabledTemplate, 
  accountReactivatedTemplate
} = require("../utils/emailTemplates");

const BASE_URL = "http://localhost:5173";
const API_URL = "http://localhost:5000";

const initiateUserDeactivation = async (userId) => {
    const user = await User.findById(userId);
    if (!user) return null;
  
    const now = new Date();
    const deactivateAt = new Date(now.getTime() + 24 * 60 * 60 * 1000);
    
    user.pendingDeactivation = true;
    user.deactivateAt = deactivateAt;
    await user.save();
  
    await sendMail(
      user.email, 
      "Account deactivation notice", 
      deactivationNoticeTemplate(`${BASE_URL}/signin`)
    );
  
    return {
      message: "User deactivation initiated. User has 24 hours to log in.",
      deactivateAt: deactivateAt
    };
};
  
const initiateUserBulkDeactivation = async (userIds) => {
    if (!Array.isArray(userIds) || userIds.length === 0) {
      return { error: "No valid user IDs provided" };
    }
    
    const now = new Date();
    const deactivateAt = new Date(now.getTime() + 24 * 60 * 60 * 1000);
    
    // Update all users to pending deactivation status
    const updateResult = await User.updateMany(
      { _id: { $in: userIds } },
      { 
        $set: { 
          pendingDeactivation: true, 
          deactivateAt: deactivateAt 
        } 
      }
    );
    
    // Send email notifications to each user
    const users = await User.find({ _id: { $in: userIds } }).select('email');
    
    for (const user of users) {
      await sendMail(
        user.email, 
        "Account deactivation notice", 
        deactivationNoticeTemplate(`${BASE_URL}/signin`)
      );
    }
    
    return { 
      modifiedCount: updateResult.modifiedCount,
      deactivateAt: deactivateAt
    };
};

let isProcessing = false; // Lock flag

const processExpiredGracePeriods = async () => {
    if (isProcessing) {
        console.log("Skipping duplicate execution of processExpiredGracePeriods...");
        return { processedCount: 0, message: "Already processing, skipping execution." };
    }

    isProcessing = true; // Set lock
    const now = new Date();

    try {
        // Find all users with expired grace periods
        const usersToDeactivate = await User.find({
            pendingDeactivation: true,
            deactivateAt: { $lt: now },
            isDeactivated: false
        });

        let processed = 0;

        for (const user of usersToDeactivate) {
            user.isDeactivated = true;
            user.pendingDeactivation = false;
            user.deactivatedAt = now;
            await user.save();

            await sendMail(
                user.email, 
                "Your account has been disabled", 
                accountDisabledTemplate(`${API_URL}/api/auth/request-reactivation?userId=${user._id}`)
            );

            processed++;
        }

        return { processedCount: processed, message: `Processed ${processed} expired grace periods` };
    } catch (error) {
        console.error("Error processing expired grace periods:", error);
        return { processedCount: 0, message: "Error processing expired grace periods" };
    } finally {
        isProcessing = false; // Release lock
    }
};
  
  const reactivateUser = async (userId) => {
    const user = await User.findById(userId);
    if (!user) return null;
  
    user.isDeactivated = false;
    user.deactivatedAt = null;
    user.pendingDeactivation = false;
    user.deactivateAt = null;
    user.hasRequestedReactivation = false;
    await user.save();
  
    await sendMail(
      user.email, 
      "Your account has been reactivated", 
      accountReactivatedTemplate(`${BASE_URL}/signin`)
    );
  
    return { message: "User reactivated successfully" };
  };
  
  module.exports = {
    initiateUserDeactivation,
    initiateUserBulkDeactivation,
    processExpiredGracePeriods,
    reactivateUser
  };
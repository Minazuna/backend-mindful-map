const deactivationNoticeTemplate = (loginUrl) => `
  <div style="background-color: #f9f9f9; padding: 20px; font-family: 'Roboto', sans-serif;">
    <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; padding: 20px; border-radius: 10px; text-align: center;">
      <img src="${process.env.FRONTEND_URL}/images/logo.png" alt="Mindful Map" style="width: 100px; margin-bottom: 20px;">
      <h2>Account Deactivation Notice</h2>
      <p style="text-align: justify;">Good day! We noticed you haven't logged in recently. Your account will be disabled in 24 hours unless you log in.</p>
      <p style="text-align: justify;">To keep your account active, please log in within the next 24 hours.</p>
      <a href="${loginUrl}" style="display: inline-block; padding: 10px 20px; margin-top: 20px; background-color: #6fba94; color: #ffffff; text-decoration: none; border-radius: 5px;">Log In</a>
    </div>
  </div>
`;

const accountDisabledTemplate = (reactivationUrl) => `
  <div style="background-color: #f9f9f9; padding: 20px; font-family: 'Roboto', sans-serif;">
    <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; padding: 20px; border-radius: 10px; text-align: center;">
      <img src="${process.env.FRONTEND_URL}/images/logo.png" alt="Mindful Map" style="width: 100px; margin-bottom: 20px;">
      <h2>Account Disabled</h2>
      <p style="text-align: justify;">Good day! Your account has been disabled due to inactivity. To reactivate your account, please send a reactivation request.</p>
      <a href="${reactivationUrl}" style="display: inline-block; padding: 10px 20px; margin-top: 20px; background-color: #6fba94; color: #ffffff; text-decoration: none; border-radius: 5px;">Send Request</a>
    </div>
  </div>
`;

const accountReactivatedTemplate = (loginUrl) => `
  <div style="background-color: #f9f9f9; padding: 20px; font-family: 'Roboto', sans-serif;">
    <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; padding: 20px; border-radius: 10px; text-align: center;">
      <img src="${process.env.FRONTEND_URL}/images/logo.png" alt="Mindful Map" style="width: 100px; margin-bottom: 20px;">
      <h2>Account Reactivated</h2>
      <p style="text-align: justify;">Good news! Your account has been reactivated by an administrator. You can now log in again and continue using Mindful Map.</p>
      <a href="${loginUrl}" style="display: inline-block; padding: 10px 20px; margin-top: 20px; background-color: #6fba94; color: #ffffff; text-decoration: none; border-radius: 5px;">Login</a>
    </div>
  </div>
`;

module.exports = {
  deactivationNoticeTemplate,
  accountDisabledTemplate,
  accountReactivatedTemplate
};

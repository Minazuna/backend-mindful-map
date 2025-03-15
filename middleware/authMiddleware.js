const jwt = require('jsonwebtoken');
const User = require('../models/User'); // Assuming you have a User model

const authMiddleware = async (req, res, next) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');

  if (!token) {
    return res.status(401).json({ success: false, message: 'No token provided. Please sign in.' });
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    console.log("Decoded Token:", decoded);

    const user = await User.findById(decoded.uid); 
    console.log("Fetched User:", user);

    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found.' });
    }

    if (user.isDeactivated) {
      return res.status(403).json({ success: false, message: "Your account has been deactivated. Please contact support." });
    }
    
    req.user = user;
    next();
  } catch (error) {
    console.error('Error in authMiddleware:', error);
    return res.status(401).json({ success: false, message: 'Invalid token. Please sign in again.' });
  }
};

const adminMiddleware = (req, res, next) => {
  if (req.user && req.user.role === 'admin') {
    next();
  } else {
    return res.status(403).json({ success: false, message: 'Access denied. Admins only.' });
  }
};

const userMiddleware = (req, res, next) => {
  if (req.user && req.user.role === 'user') {
    next();
  } else {
    return res.status(403).json({ success: false, message: 'Access denied. Users only.' });
  }
};

module.exports = { authMiddleware, adminMiddleware, userMiddleware };
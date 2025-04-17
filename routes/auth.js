const express = require('express');
const router = express.Router();
const passport = require('passport');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const bcrypt = require('bcryptjs');
const { Strategy: JwtStrategy, ExtractJwt } = require("passport-jwt");

const opts = {
  jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
  secretOrKey: process.env.JWT_SECRET, // make sure it's defined
};

passport.use(
  new JwtStrategy(opts, async (jwt_payload, done) => {
    try {
      const user = await User.findById(jwt_payload.id);
      if (user) return done(null, user);
      else return done(null, false);
    } catch (err) {
      return done(err, false);
    }
  })
);

router.use(passport.initialize());

// Helper functions
const generateToken = (user) => jwt.sign(
  { id: user.id, method: user.method },
  process.env.JWT_SECRET,
  { expiresIn: '7d' }
);

const filterUser = (user) => ({
  id: user.id,
  email: user.email,
  username: user.username,
  displayName: user.displayName,
  avatar: user.avatar,
  method: user.method
});

// Middleware to protect routes
const requireAuth = passport.authenticate('jwt', { session: false });

// Local signup
router.post('/signup', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    if (!email || !password) {
      return res.status(400).json({ message: 'Email and password required' });
    }

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: 'Email already exists' });
    }

    const newUser = await User.create({
      method: 'local',
      email,
      password
    });

    const token = generateToken(newUser);
    res.status(201).json({ token, user: filterUser(newUser) });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// Local login
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email, method: 'local' });
    
    if (!user) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    const token = generateToken(user);
    res.json({ token, user: filterUser(user) });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});


// backend/routes/auth.js
router.post('/google', async (req, res) => {
  const { token } = req.body;

  try {
    const ticket = await client.verifyIdToken({
      idToken: token,
      audience: process.env.GOOGLE_CLIENT_ID,
    });

    const payload = ticket.getPayload();

    // Example: Check if user exists, create user, generate JWT
    const user = await findOrCreateUser(payload);
    const jwtToken = generateYourAppToken(user);

    console.log("Yes");
    res.json({ token: jwtToken });
  } catch (err) {
    console.error('Google token verification failed:', err);
    res.status(401).json({ error: 'Invalid token' });
  }
});


router.get('/google/callback', 
  passport.authenticate('google', { session: false }),
  (req, res) => {
    const token = generateToken(req.user);
    res.redirect(`/auth/success?token=${token}&user=${JSON.stringify(filterUser(req.user))}`);
  }
);

// ===== NEW PROFILE MODIFICATION ROUTES ===== //

// Change username (protected route)
router.patch('/update-username', requireAuth, async (req, res) => {
  try {
    const { username } = req.body;
    const userId = req.user.id;

    if (!username || username.length < 4) {
      return res.status(400).json({ message: 'Username must be at least 4 characters' });
    }

    // Check if username is already taken
    const existingUser = await User.findOne({ username });
    if (existingUser && existingUser.id !== userId) {
      return res.status(400).json({ message: 'Username already taken' });
    }

    const updatedUser = await User.findByIdAndUpdate(
      userId,
      { username },
      { new: true }
    );

    res.json({ 
      message: 'Username updated successfully',
      user: filterUser(updatedUser)
    });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// Change email (protected route)
router.patch('/update-email', requireAuth, async (req, res) => {
  try {
    const { email, password } = req.body;
    const userId = req.user.id;

    if (!email || !password) {
      return res.status(400).json({ message: 'Email and password required' });
    }

    // Verify password first
    const user = await User.findById(userId);
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ message: 'Invalid password' });
    }

    // Check if email is already in use
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: 'Email already in use' });
    }

    const updatedUser = await User.findByIdAndUpdate(
      userId,
      { email, verified: false }, // Mark as unverified after email change
      { new: true }
    );

    // In a real app, you would send a verification email here
    res.json({ 
      message: 'Email updated successfully. Please verify your new email.',
      user: filterUser(updatedUser)
    });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

router.post('/reset-password', async (req, res) => {
  const { email, newPassword } = req.body;
  
  try {
    // 1. Find user by email
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(404).json({ success: false, message: "User not found" });
    }
    
    
    // 3. Update user's password
    user.password = newPassword;
    await user.save();
    
    // 4. Return success
    res.json({ success: true, message: "Password updated successfully" });
  } catch (error) {
    console.error("Password reset error:", error);
    res.status(500).json({ success: false, message: "Failed to reset password" });
  }
});

// Get current user profile (protected)
router.get('/me',passport.authenticate("jwt", { session: false }), async (req, res) => {
  const user = await User.findById(req.user.id);
  res.json({ user: filterUser(user) });
});


module.exports = router;
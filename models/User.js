const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const UserSchema = new mongoose.Schema({
  email: { type: String, unique: true },
  password: String,
  username: { type: String, unique: true },
  googleId: String,
  displayName: String,
  avatar: String,
  method: { type: String, enum: ['local', 'google'], required: true }
});

// Generate username from email
UserSchema.pre('save', function(next) {
  if (!this.username) {
    if (this.method === 'local') {
      this.username = this.email.split('@')[0] + 
                      Math.floor(1000 + Math.random() * 9000);
    } else if (this.method === 'google') {
      this.username = this.displayName.replace(/\s+/g, '').toLowerCase() + 
                      Math.floor(1000 + Math.random() * 9000);
    }
  }
  next();
});

// Hash password before saving
UserSchema.pre('save', async function(next) {
  try {
    if (this.method !== 'local') return next();
    
    if (!this.isModified('password')) return next();
    
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
  } catch (err) {
    next(err);
  }
});

module.exports = mongoose.model('User', UserSchema);
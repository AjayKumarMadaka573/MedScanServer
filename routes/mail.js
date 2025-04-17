require("dotenv").config();
const express = require("express");
const nodemailer = require("nodemailer");

const router = express.Router();
router.use(express.json());

console.log("acc"+process.env.SMTP_USER)
// 🔐 Fixed recipient email (e.g., admin/support)
const RECEIVER_EMAIL = "accforbusy573@gmail.com"; // <- change this to your constant email

const transporter = nodemailer.createTransport({
  host: process.env.SMTP_HOST,
  port: process.env.SMTP_PORT,
  secure: false, // true for 465, false for 587
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS,
  },
});

// 🔗 POST /send-mail
router.post("/send-mail", async (req, res) => {
  const { username, message } = req.body;
  const subject = `Query from ${username}`;
  try {
    const info = await transporter.sendMail({
      from: `"MedScan" <${process.env.SMTP_USER}>`,
      to: RECEIVER_EMAIL,
      subject,
      html: `
        <p><strong>Message from:</strong> ${username}</p>
        <p><strong>Message:</strong><br/>${message}</p>
      `,
    });

    res.json({ success: true, messageId: info.messageId });
  } catch (err) {
    console.error("Email sending failed:", err);
    res.status(500).json({ success: false, error: "Failed to send email." });
  }
});

module.exports = router;

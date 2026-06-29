# PayVerify Nepal — Setup Guide (v2 with Login & Dashboard)

## NEW FEATURES ADDED
- User Login / Signup (per-shop accounts)
- Dashboard with charts (trend, verdict breakdown, platform breakdown)
- About, How It Works, and Contact pages
- Mobile-responsive navigation
- Animations and visual polish throughout

## RUNNING THE APP
1. Open VS Code in the payverify/ folder
2. Terminal: pip install -r requirements.txt
3. Terminal: python app.py
4. Open http://localhost:5000

## FIRST TIME USE
1. Click "Sign Up" in the navbar
2. Enter a shop name, username, password
3. You're logged in! Now upload screenshots from the Home page
4. Visit /dashboard to see charts of your verification history

## IMPORTANT FILES CREATED AUTOMATICALLY
- users.json      -> stores user accounts (hashed passwords)
- history.json    -> stores all verification results (per user)
- uploads/        -> stores uploaded screenshots

## TRAINING THE AI MODEL (same as before)
1. Put screenshots in dataset/genuine/ and dataset/fake/
2. python create_dataset.py
3. python train_model.py
4. Restart: python app.py

## DEPLOY TO RENDER.COM (same as before)
1. Push to GitHub
2. Connect repo on render.com
3. render.yaml + packages.txt handle config automatically
4. Note: on Render, uploads/users.json/history.json will reset on redeploy
   (free tier has no persistent disk) - mention this limitation in your thesis
   as a "future work" item, or upgrade to a paid plan with persistent disk.

## TESTING CHECKLIST FOR THESIS
- [ ] Sign up creates a new account
- [ ] Login works with correct credentials
- [ ] Login fails with wrong password (shows error)
- [ ] Upload screenshot while logged out -> redirected to login
- [ ] Upload screenshot while logged in -> works, shows result
- [ ] Dashboard shows correct counts and charts
- [ ] History table shows only YOUR verifications
- [ ] About / How It Works / Contact pages load correctly
- [ ] Mobile menu (hamburger) works on narrow screens

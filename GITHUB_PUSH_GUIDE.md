# GitHub Push Instructions

This guide will help you push your project to GitHub.

## Step 1: Create a GitHub Repository

1. Go to [github.com](https://github.com) and log in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the details:
   - **Repository name**: `sperm-quantification` (or your preferred name)
   - **Description**: "Research-grade Python pipeline for analyzing sperm motility from videomicroscopy"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 2: Link Your Local Repository

GitHub will show you commands. Use these:

```bash
# Add GitHub as remote origin (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/sperm-quantification.git

# Verify the remote was added
git remote -v
```

## Step 3: Push Your Code

```bash
# Push to GitHub (main branch)
git push -u origin main
```

If you're using an older Git version that defaults to "master":
```bash
git branch -M main
git push -u origin main
```

## Step 4: Update Repository URL in README

After creating your repository, update the URL in `README.md`:

1. Open `README.md`
2. Find: `url = {https://github.com/yourusername/sperm-quantification}`
3. Replace with your actual GitHub URL

Then commit the change:
```bash
git add README.md
git commit -m "Update repository URL in citation"
git push
```

## Troubleshooting

### Authentication Error

If you get an authentication error, you'll need to use a Personal Access Token (PAT):

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Use the token as your password when prompted

Or use SSH (recommended):
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
```

Then change remote to SSH:
```bash
git remote set-url origin git@github.com:USERNAME/sperm-quantification.git
```

## What Gets Pushed

✅ All source code (`src/`)
✅ Configuration files (`configs/`)
✅ Documentation (`README.md`, `NEXT_STEPS.md`)
✅ Example notebooks (`notebooks/`)
✅ Demo scripts (`demo.py`, `app.py`, `visualize.py`)
✅ Tests (`tests/`)
✅ Requirements (`requirements.txt`, `setup.py`)
✅ License (`LICENSE`)

❌ Data files (excluded by `.gitignore`)
❌ Cached files (`__pycache__`, `.pyc`)
❌ Virtual environments
❌ IDE settings
❌ Generated visualizations

## After Pushing

### Add Topics/Tags

On your GitHub repository page:
1. Click the gear icon next to "About"
2. Add topics: `biophysics`, `computer-vision`, `sperm-analysis`, `python`, `microfluidics`, `kalman-filter`, `who-guidelines`, `streamlit`

### Create a Nice README Badge

Add this to the top of your README:
```markdown
![GitHub stars](https://img.shields.io/github/stars/USERNAME/sperm-quantification?style=social)
![GitHub forks](https://img.shields.io/github/forks/USERNAME/sperm-quantification?style=social)
```

### Enable GitHub Pages (Optional)

To host documentation:
1. Go to repository Settings → Pages
2. Select source: `main` branch, `/docs` folder
3. Create a `docs` directory with documentation

## Next Steps

Consider:
- ⭐ Adding screenshots to README
- 📝 Writing a detailed CONTRIBUTING.md
- 🐛 Setting up issue templates
- 🔄 Creating a CHANGELOG.md
- 📊 Adding GitHub Actions for automated testing
- 🎬 Recording a demo video

## Share Your Project

Once pushed, share on:
- LinkedIn with #biophysics #python #research
- Twitter/X with relevant hashtags
- Reddit (r/Python, r/bioinformatics)
- ResearchGate

---

**Your project is now ready for the world! 🚀**

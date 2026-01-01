# Git Branching Strategy - Professional Workflow

## 📋 Branch Structure

```
main (production)
  ↑
develop (integration)
  ↑
feature/phase-X (feature branches)
```

---

## 🌳 Branch Descriptions

### `main` - Production Branch

- **Purpose:** Stable, production-ready code only
- **Protection:** Never commit directly (except initial setup)
- **Updates:** Only via Pull Requests from `develop`
- **Status:** Always deployable

### `develop` - Integration Branch  

- **Purpose:** Active development integration
- **Protection:** Tested features only
- **Updates:** Merge from `feature/*` branches after testing
- **Status:** Latest features, should be stable

### `feature/*` - Feature Branches

- **Purpose:** Individual feature development
- **Naming:** `feature/phase-4-training`, `feature/add-visualization`
- **Lifespan:** Temporary (delete after merge)
- **Updates:** Branch from `develop`, merge back to `develop`

---

## 🚀 Recommended Workflow

### For Each New Phase/Feature

```bash
# 1. Start from develop
git checkout develop
git pull origin develop

# 2. Create feature branch
git checkout -b feature/phase-4-training

# 3. Develop & commit
git add .
git commit -m "feat(training): implement trainer class"

# 4. Push feature branch
git push -u origin feature/phase-4-training

# 5. After testing: Merge to develop
git checkout develop
git merge feature/phase-4-training

# 6. Push develop
git push origin develop

# 7. Delete feature branch (cleanup)
git branch -d feature/phase-4-training
git push origin --delete feature/phase-4-training

# 8. When ready for release: Merge develop → main
git checkout main
git merge develop
git push origin main
git tag -a v1.0.0 -m "Phase 1-3 complete"
git push origin v1.0.0
```

---

## ✅ Current Setup (After running script)

- ✅ `main` branch exists (production-ready Phases 0-3)
- ✅ `develop` branch created (active development)
- ✅ Both pushed to GitHub

---

## 🛡️ Safety Rules

### DO

- ✅ Always branch from `develop` for new features
- ✅ Test thoroughly before merging
- ✅ Use Pull Requests for code review (optional solo project)
- ✅ Keep commits atomic and well-messaged
- ✅ Push `develop` frequently (backup)

### DON'T

- ❌ Never force push to `main` or `develop`
- ❌ Never commit directly to `main` (except emergencies)
- ❌ Never merge untested code to `develop`
- ❌ Never delete branches before they're merged

---

## 📊 Branch Strategy Benefits

| Risk | Without Branches | With Branches |
|------|------------------|---------------|
| **Breaking main** | High (direct commits) | Low (merge after test) |
| **Lost work** | Medium (overwrite) | Low (separate branches) |
| **Collaboration** | Conflicts | Clean merges |
| **Rollback** | Difficult | Easy (revert merge) |

---

## 🎯 Next Steps (Phase 4)

```bash
# Start Phase 4 on feature branch
git checkout -b feature/phase-4-training

# Develop trainer, evaluation, etc.
# ... coding ...

# Commit progress
git add .
git commit -m "feat(training): add trainer class with MLflow"

# Push for backup
git push -u origin feature/phase-4-training

# After Phase 4 complete:
git checkout develop
git merge feature/phase-4-training
git push origin develop

# When ALL phases done (ready for release):
git checkout main
git merge develop  # Merge all phases to production
git tag -a v1.0.0 -m "Full RL-CDSS implementation"
git push origin main --tags
```

---

## 🔒 GitHub Branch Protection (Optional - Set via GitHub UI)

**Go to:** Settings → Branches → Add Rule

**For `main` branch:**

- ✅ Require pull request before merging
- ✅ Require status checks to pass (CI/CD)
- ✅ Require linear history
- ✅ Include administrators (even you!)

**For `develop` branch:**

- ✅ Require pull request (optional for solo)
- ✅ Require status checks

---

## 📝 Quick Reference

**Check current branch:**

```bash
git branch
```

**Switch branch:**

```bash
git checkout develop
# or
git checkout feature/phase-4-training
```

**Create & switch:**

```bash
git checkout -b feature/new-feature
```

**Merge feature to develop:**

```bash
git checkout develop
git merge feature/phase-4-training
```

**Delete branch (after merge):**

```bash
git branch -d feature/phase-4-training  # Local
git push origin --delete feature/phase-4-training  # Remote
```

---

## ✅ Safety Checklist

Before merging to `develop`:

- [ ] All tests passing
- [ ] Code reviewed (self-review minimum)
- [ ] Commit messages clear
- [ ] No merge conflicts

Before merging `develop` to `main`:

- [ ] All phases complete and tested
- [ ] Documentation updated
- [ ] README accurate
- [ ] Ready for public/portfolio showcase

---

**Current Status:**

- ✅ `main`: Phases 0-3 (stable)
- ✅ `develop`: Created (active development)
- ⏳ Next: Create `feature/phase-4-training` for Phase 4

**Protection Enabled:** Development isolated from production ✅

# project-common
motive of this repo to create a easy and effective project deployment



Great! Let’s go step-by-step to save your Git learning notes as a file in your GitHub repo:

---

### ✅ **Step 1: Make a Folder to Store Learning Notes**

Go to your project directory (already in `project-common` folder):

```bash
mkdir git-learn-notes
```

This creates a new folder named `git-learn-notes` inside your project.

---

### ✅ **Step 2: Create the Markdown File**

Now open a file called `my_git_learning.md` inside that folder:

```bash
nano git-learn-notes/my_git_learning.md
```

You'll see a blank screen. **Copy and paste the below full content**:

---

### 📄 Paste the following in `nano`:

````markdown
# 🧠 My Git & GitHub Learning: Full Explanation

This file captures what I understood when I connected my local project to GitHub and pushed my code. This is based on my learning journey using Linux and Git for the first time.

---

## Step 1: Installing Git

Since I was using Ubuntu (Linux), the first step was to make sure Git was installed. I ran:

```bash
sudo apt install git
````

> This command installs Git if it’s not already installed. Git is the tool that helps track code changes and allows communication between my computer and GitHub.

---

## Step 2: Cloning a GitHub Repository

I created a new repository on GitHub called `project-common`. Then, I cloned it to my system using:

```bash
git clone https://github.com/thanush-robin/project-common.git
```

> This created a new folder named `project-common` in my local system, which is now connected to the GitHub repo. This folder acts like a **bridge** between GitHub and my computer.

---

## Step 3: Moving My Existing Project Files into the Cloned Repo

This step was a bit tricky for me to understand at first.

My actual local project files were stored in a parent folder called `Thanush_new_project_v1`, and inside that folder, the `project-common` folder was created after cloning.

So my goal was to **move all files from `Thanush_new_project_v1`** (except the `project-common` folder itself) **into the `project-common` folder**.

That’s why this command was used:

```bash
shopt -s extglob  # enables advanced pattern matching
mv ../!(project-common) ./ 2>/dev/null
```

> `mv` stands for move.
> `!(project-common)` means “move everything except the folder named `project-common`.”
> `./` means move them *into the current directory*, which is `project-common`.

Now all my project files are inside the Git-tracked folder and ready to be committed.

---

## Step 4: Setting Up Git Identity (One Time Only)

Before committing any changes, Git asked me to tell who I am. I used my GitHub name and email:

```bash
git config --global user.name "Thanush Bharat"
git config --global user.email "thanushbharath186@gmail.com"
```

> This lets Git label my commits correctly, and GitHub can show that the code was pushed by me.

---

## Step 5: Staging and Committing Changes

Once the files were in the `project-common` folder, I told Git to start tracking all of them and then committed the changes:

```bash
git add .
git commit -m "Initial upload: moved project files into repo"
```

* `git add .` tells Git: “Track all the changes and new files in this folder.”
* `git commit -m "..."` creates a snapshot with a message. This is like saving a version of my project.

---

## Step 6: Generating a GitHub Token & Pushing the Code

GitHub no longer allows password authentication when pushing code from the terminal. So I generated a **Personal Access Token (Classic)** from this link:

👉 [https://github.com/settings/tokens](https://github.com/settings/tokens)

### Permissions I selected:

* ✅ `repo` — Full control over repositories (enough to push code)
* ✅ `public_repo` — If your project is public

Then I pushed my code using:

```bash
git push origin main
```

Git asked for:

* Username → I entered: `thanush-robin`
* Password → I **pasted the generated token** (not my GitHub password)

> After this, the code was uploaded (pushed) to GitHub and became available online.

---

## 🔁 Summary of What I Learned

* How to install and use Git
* How `git clone` connects local system to GitHub
* The meaning of `mv` to move files safely into the Git repo
* How to configure Git with name and email
* The purpose of `git add` and `git commit`
* How to push to GitHub using a token

This is now a skill I can use in every project from now on.

---

📌 I'm saving this as a learning note in my project so I can refer to it anytime or help someone else learn from it too.

— **Thanush Bharat**

````

---

### ✅ **Step 3: Save and Exit**

1. Press `Ctrl + O` → then press `Enter` to save.
2. Press `Ctrl + X` to exit the editor.

---

### ✅ Step 4: Commit and Push to GitHub

Now run:

```bash
git add git-learn-notes/my_git_learning.md
git commit -m "Added Git learning notes"
git push origin main
````

---

Once you’ve done this, go to your GitHub repo and you’ll see the `git-learn-notes` folder and your personal `my_git_learning.md` file inside it ✅

Let me know when you're done, or if you’d like a template for documenting your **Python learning** too!


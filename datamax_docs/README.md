Of course. Here is the English translation of the maintenance manual, with formatting preserved.

-----

# DataMax Project Documentation Maintenance Manual

> This is the internal maintenance manual for the official DataMax project documentation website. The site is built with [Docusaurus](https://docusaurus.io/).

**Live Site URL: [https://Hi-Dolphi.github.io/datamax/](https://Hi-Dolphi.github.io/datamax/)**

## ✨ Core Features

  * **Rich Content**: Covers everything from quick starts to advanced features.
  * **Full-Text Search**: Fast full-text search powered by Algolia.
  * **Versioning**: Supports multi-version documentation management.
  * **Dark/Light Mode**: Automatically adapts to the system's appearance preferences.

-----

## 🚀 Local Development

Team members should run the site locally before modifying or adding any content to preview changes in real-time.

### 1\. Prerequisites

  * [Node.js](https://nodejs.org/en/) (version \>= 18.0)
  * [NPM](https://www.npmjs.com/) (installed automatically with Node.js)

You can check your versions by running the following commands in your terminal:

```bash
node -v
npm -v
```

### 2\. Installation & Running

1.  Clone this repository to your local machine.

2.  Navigate to the Docusaurus project directory (the website's code is located in the `datamax_docs` subdirectory):

    ```bash
    cd datamax/datamax_docs
    ```

    **Note: All subsequent commands must be executed from this directory.**

3.  Install project dependencies (run this for the first time or after dependencies have been updated):

    ```bash
    npm install
    ```

4.  Start the local development server:

    ```bash
    npm run start
    ```

    After executing, the website will be running at `http://localhost:3000`. This service supports hot reloading, so when you modify and save a file, the browser will automatically refresh for real-time previews.

-----

## 📝 Content Maintenance Guide

This section provides detailed daily maintenance instructions for team members.

### Project Structure Overview

The files and folders most relevant to our daily work are located within `datamax_docs`:

```
datamax_docs/
├── docs/               <-- ✅ Location for all documentation Markdown files
├── static/
│   └── img/            <-- ✅ Location for static assets like images
├── src/
│   └── css/
│       └── custom.css  <-- ✅ Customize website styles here
├── sidebars.js         <-- ❗️ Sidebar structure definition file
└── docusaurus.config.js  <-- ❗️ Core configuration file (site title, navbar, etc.)
```

### Modifying and Adding Documents

  * **To modify a document**: Simply find the corresponding Markdown file in the `docs/` folder, open it with any text editor (VS Code is recommended), and make your changes.
  * **To add a new document**: Create a new `.md` file within the `docs/` folder or one of its subfolders.

### How Does the Sidebar Work? (Autogenerated)

Our sidebar is configured in **autogenerated mode**.

This means:

  * **You do not need to manually edit the `sidebars.js` file.**
  * Simply **add, delete, or rename** Markdown files or folders within the `docs` directory, and the sidebar will **update automatically**.
  * The sidebar's order and grouping structure will **automatically map to the file and folder structure of the `docs` directory**, typically arranged alphabetically.

-----

## 🎨 Advanced Configuration

When you need to modify the website's appearance or overall structure, edit the following files:

  * **Site Metadata** (title, logo, navbar, etc.): `docusaurus.config.js`
  * **Website Theme Colors**: `src/css/custom.css`

Modifying these files requires some knowledge of Docusaurus. Please proceed with caution or refer to the official documentation.

-----

## Deployment

This website is automatically deployed via **GitHub Actions**. Any commit pushed to the `main` branch will trigger an automatic build and release process, deploying the latest documentation to GitHub Pages.

**Team members only need to push their changes to the `main` branch; no manual deployment commands are necessary.**

-----

*This documentation is powered by Docusaurus.*
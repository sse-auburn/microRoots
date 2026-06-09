# Uploading This Package to GitHub from the Browser

This package is designed so you can upload it to GitHub without using Git, WSL, or command line tools.

## Steps

1. Download `microRoots_repo.zip`.
2. Right-click the ZIP file on Windows and select **Extract All**.
3. Open the extracted folder.
4. Open your GitHub repository in the browser.
5. Click **Add file**.
6. Click **Upload files**.
7. Drag all extracted files and folders into the upload area.
8. Add the commit message:

```text
Add microRoots pipeline code and documentation
```

9. Click **Commit changes**.

## Do not upload weights

Do not upload these files to GitHub:

```text
last.pt
FinalFT.pt
```

They are large model weights and should stay on Google Drive.

The repository only includes instructions telling users where to download the weights.

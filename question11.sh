git branch branch2
git switch branch2
touch file4
git add .
git commit -m "file4"
echo "123"> ./file4
git stash
git switch main
apt-get update && apt-get install -y zsh && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

apt-get install -y git-lfs vim

pip install uv
uv sync
uv venv

python -m spacy download en_core_web_sm
git lfs install

git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Add zsh-autosuggestions to the plugins list
if ! grep -q "zsh-autosuggestions" ~/.zshrc; then
    sed -i 's/^plugins=(.*)/plugins=(\1 zsh-autosuggestions)/' ~/.zshrc
fi

# Add zsh-syntax-highlighting to the plugins list
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# First check if it's not already in the plugins
if ! grep -q "zsh-syntax-highlighting" ~/.zshrc; then
    # Remove the closing parenthesis, add the new plugin, and add the closing parenthesis back
    sed -i 's/plugins=(/plugins=(zsh-syntax-highlighting /' ~/.zshrc
fi

bash prepare/download_pretrained_models.sh
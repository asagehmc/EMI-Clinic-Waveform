In order to run the mesa pipeline (mesa_pipeline.py), a few files need to added that are too large to store in the
repo on their own.

To run the code, please download the following files:
    ApproximateNetwork.h5 https://drive.google.com/file/d/1R0t3VxPLBpQmIKKyH9ulDdr9irwsyOpD/view?usp=sharing
    RefinementNetwork.h5 https://drive.google.com/file/d/1qc97paeXHDWrOEsPR1_2tl7WtMcJxQbF/view?usp=sharing
Place these files in the directory PPGtoBP/PPG_model/bloodPressureModel/models

--------------------------------------------------

To install the ruby gem for nsrr (on mac):

brew install rbenv ruby-build
echo 'eval "$(rbenv init -)"' >> ~/.zshrc
source ~/.zshrc
rbenv install 3.2.2
rbenv global 3.2.2
gem install nsrr

If it still does not work, it may be using an outdated version of Ruby. Run this:
echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(rbenv init -)"' >> ~/.zshrc
source ~/.zshrc

--------------------------------------------------

You will also need a token.txt file to run the mesa download. This file should be a 1-line file containing only the NSRR
token which can be accessed by logging in at
    https://sleepdata.org/token
Place this file in the PPGtoBP directory (same level as this README)


--------------------------------------------------

If not on Mac, you might need to add a custom path to the gem for download. To do this, find the gem path by running:
    ruby -e "puts Gem.bindir"
Then run mesa_pipeline with the gem path as the single argument (with "nsrr.bat" appended to the end.) It
The path should look something like this:
    C:/Ruby34-x64/bin/nsrr.bat

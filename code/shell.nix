with import <nixpkgs> {};

stdenv.mkDerivation {
    name = "poetry";
    buildInputs = [
      fish
      git
      python311
      poetry
    ];
    shellHook = ''
        export ENVNAME=poetry
        fish --init-command="function fish_greeting; echo 'Entered $ENVNAME Environment'; end; function fish_prompt; echo '$ENVNAME 🐟> '; end;"
        echo "Leaving $ENVNAME Environment"
        exit
    '';
}

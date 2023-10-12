{
    inputs = {
	nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
	flake-utils.url = "github:numtide/flake-utils";
	rust-overlay = {
	    url = "github:oxalica/rust-overlay";
	    inputs = {
		nixpkgs.follows = "nixpkgs";
		flake-utils.follows = "flake-utils";
	    };
	};
	crane = {
	    url = "github:ipetkov/crane";
	    inputs = {
		nixpkgs.follows = "nixpkgs";
		rust-overlay.follows = "rust-overlay";
		flake-utils.follows = "flake-utils";
	    };
	};
    };

    outputs = { self, nixpkgs, flake-utils, rust-overlay, crane }:
	flake-utils.lib.eachDefaultSystem
	    (system:
		let
		    overlays = [ (import rust-overlay) ];
		    pkgs = import nixpkgs {
			inherit system overlays;
		    };
		    rustToolchain = pkgs.pkgsBuildHost.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
		    craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
		    src = craneLib.cleanCargoSource ./.;

		    nativeBuildInputs = with pkgs; [ rustToolchain pkg-config ];
		    buildInputs = with pkgs; [
			gfortran
			(pkgs.lib.getLib gfortran.cc)
			openblas
			openssl
			cpp-netlib
			];
		    commonArgs = {
			inherit src buildInputs nativeBuildInputs;
			pname = "error_margin";
			version = "0.1.0";
		    };
		    cargoArtifacts = craneLib.buildDepsOnly commonArgs;

		    core = craneLib.buildPackage (commonArgs // {
			inherit cargoArtifacts;
			pname = "error_margin";
		    });

		in
		with pkgs;
		{
		    packages =
			{
			    inherit core;
			    default = core;
			};
		    devShells.default = mkShell {
			inputsFrom = [ core ];
			shellHook = ''
			    if ! test -d .nix-shell; then
            		      mkdir .nix-shell
            		    fi

            		    export NIX_SHELL_DIR=$PWD/.nix-shell
			'';

		    };
		}
	    );
}

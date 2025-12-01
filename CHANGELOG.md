# Change Log
All notable changes to this project will be documented in this file.

## [1.1.3]
Minor bugfixes. 
1. Restrict numpy version to less than 2 in pyproject.toml.
2. Exception handling when checking video files

Minor feature add. Export annotation overlaid on video from the VideoAnnotation class.

## [1.1.2]
Minor bugfix. Fixed error in the palette code when seaborn is not installed
Minor feature add. Export h5 files as json.

## [1.1.1]
Minor bugfix related to adding annotation layers after creating the VideoPointAnnotator interface.

## [1.1.0]

### Changed
Removed the limitation of being able to annotate a maximum of 10 labels. Now, the new limit is 1000 labels. 

## [1.0.2] - 2025-05-07
 
First major release.

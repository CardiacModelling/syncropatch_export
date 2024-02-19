
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<!-- <br /> -->
<!-- <div align="center"> -->
<!--     <img src="images/logo.png" alt="Logo" width="80" height="80"> -->
<!--   </a> -->


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This project contains a python package and scripts for extracting data from highthroughput patch-clamp experiments such as those performed on a Nanion SynroPatch 384. The package contains a lot of useful functionality that can be used to perform:
- leak correction;
- QC checks;
- plotting trace;
- exporting data into different file.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This package has been tested on Ubuntu with Python3.7 Python3.8 Python3.9 Python3.10 and Python3.11.

### Installation

irst clone this repository

```
git clone git@github.com:CardiacModelling/syncropatch_export && cd syncropatch_export
```

With one of these versions install, create and activate a virtual environment.

  ```sh
  python3 -m venv .venv && source .venv/bin/activate
  ```

Then install the package with `pip`.

```
python3 -m pip install --upgrade pip && python3 -m pip install -e .'[test]'
```

To run the tests you must first download some test data. Test data is available at [cardiac.nottingham.ac.uk/syncropatch\_export](https://cardiac.nottingham.ac.uk/syncropatch_export)

```
wget https://cardiac.nottingham.ac.uk/syncropatch_export/test_data.tar.xz -P tests/
tar xvf tests/test_data.tar.xz
```

Then you can run the tests.
```
python3 -m unittest
```

<!-- USAGE EXAMPLES -->
## Usage

TODO

<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Joseph Shuttleworth joseph.shuttleworth@nottingham.ac.uk

Project Link: [https://github.com/CardiacModelling/syncropatch\_export](https://github.com/CardiacModelling/syncropatch_export)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/CardiacModelling/syncropatch_export.svg?style=for-the-badge
[contributors-url]: https://github.com/CardiacModelling/syncropatch_export/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/CardiacModelling/syncropatch_export.svg?style=for-the-badge
[forks-url]: https://github.com/CardiacModelling/syncropatch_export/network/members
[stars-shield]: https://img.shields.io/github/stars/CardiacModelling/syncropatch_export.svg?style=for-the-badge
[stars-url]: https://github.com/CardiacModelling/syncropatch_export/stargazers
[issues-shield]: https://img.shields.io/github/issues/CardiacModelling/syncropatch_export.svg?style=for-the-badge
[issues-url]: https://github.com/CardiacModelling/syncropatch_export/issues
[license-shield]: https://img.shields.io/github/license/Cardiac/Modelling/syncropatch_export.svg?style=for-the-badge
[license-url]: https://github.com/CardiacModelling/syncropatch_export/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

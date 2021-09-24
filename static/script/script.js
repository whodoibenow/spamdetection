













/**************** Scroll Indicator ******************/

let progressSection = document.querySelector('.progress-section');
let progressBar = document.querySelector('.progress-bar');
let progressNum = document.querySelector('.progress-num');



function updateProgressBar()
{

    progressBar.style.height = `${getScrollPercentage()}%`
  
    requestAnimationFrame(updateProgressBar)
}

function getScrollPercentage()
{
    return (100-window.scrollY/(document.body.scrollHeight - window.innerHeight)*100);
}

updateProgressBar()


/*************** Hamburger Toggle****************/

const hamburger = document.querySelector(".hamburger");
const navMenu = document.querySelector(".nav-list");

hamburger.addEventListener("click", mobileMenu);

function mobileMenu() {
    hamburger.classList.toggle("active");
    navMenu.classList.toggle("active");
}


/******************** Webcam Toggle ***********************/

function toggle() {
    var element = document.getElementById('videodisplay');
    var change = document.getElementById("button");
    if(element.style.display === "none" && change.value=="CHECK" )
    {
        element.style.display = "block";
        change.value = "CLOSE";
        
    }
    else
    {
        
        element.style.display = "none";
        change.value = "CHECK";
    }
}

function click(){
    var toggle = document.getElementById("button")
    if(toggle.value == "CHECK")
    {
        toggle.value = "CLOSE";
    }
    else
    {
        toggle.value = "CHECK";
    }
}

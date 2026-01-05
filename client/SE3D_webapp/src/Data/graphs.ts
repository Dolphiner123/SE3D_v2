import type {SpectrumGraph} from "../Types/spectrumGraph"

export const graphData:SpectrumGraph[] = [

    {
        id: 1,
        title: "SE3D",
        subheading: "hello :D",
        sliderValues: [10.5,-3,45,0.55,0.65,5.1,2.05,0.6,5.1,0.5,7.0,6.8,0,7.0,0,0.252],
        sliderRanges: [[9.5,11.5],[-4.0,-2.0],[0,90],[0.3,1.0],[0.1,1.0],[0.2,10],[0.1,4.0],[0.1,1.0],[0.2,10],[0,1],[1,12.6],[0.1,13.9],[-0.3,0.3],[0.1,13.9],[-0.3,0.3],[0.004,0.5]],
        sliderIcons: ["MStar.png","MDust.png","Theta.png","RStar.png","CStar.png","NStar.png","RDust.png","CDust.png","NDust.png","Clump.png","Lookback.png","Lookback.png","Lookback.png","Lookback.png","Lookback.png",""],
        sliderOrderOnWebsite: ["MStar","MDust","Theta","RStar","CStar","NStar","RDust","CDust","NDust","Clump","Lookback1","Lookback2","Lookback3","Lookback4","Lookback5","slider16"],
        TelescopeIcons: ["Hubble.jpeg","JamesWebb.jpg","Herschel.jpeg","Alma.jpeg"],
        TelescopeNames: ["HST","JWST","Herschel","ALMA"],
        axesRange: [[-10,10],[-10,10],[-10,10],[-100,100],[-1000,1000],[-10,10],[-1000,1000]],
        helpTexts: ["Changes dust mass","Changes stellar mass","Changes galaxy size","","","",""],
        maxSlidersPerPage: 8
    }
    
]

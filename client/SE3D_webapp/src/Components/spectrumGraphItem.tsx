import type {SpectrumGraph} from "../Types/spectrumGraph"
import { useState, useMemo, useEffect} from "react"
import { Line } from "react-chartjs-2";
//import { Chart, LineController, LineElement, PointElement, LinearScale, Title, CategoryScale } from "chart.js";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LogarithmicScale,  
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  LineController,
} from "chart.js";


ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,   
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

ChartJS.register(LineController, LineElement, PointElement, LinearScale, Title, CategoryScale);

interface SpectrumGraphItemProps{
    spectrumGraph:SpectrumGraph
}

export default function SpectrumGraphItem({spectrumGraph}:SpectrumGraphItemProps){

   //                                                                 -----     GLOBAL VARIABLES     ----- 

    const [sliderValues, setSliderValues] = useState([...spectrumGraph.sliderValues]);
    useEffect(() => {
        setSliderValues([...spectrumGraph.sliderValues]);
    }, [spectrumGraph.sliderValues]);

    const [currentSliderPage, setCurrentSliderPage] = useState(0)
    let lastSliderPage = Math.floor(sliderValues.length / spectrumGraph.maxSlidersPerPage)
    if (sliderValues.length % spectrumGraph.maxSlidersPerPage == 0){
        lastSliderPage--
    }
    const default_values = spectrumGraph.sliderValues
    const slidersOnPage = sliderValues.slice(currentSliderPage*spectrumGraph.maxSlidersPerPage,Math.min(sliderValues.length, (currentSliderPage+1)*(spectrumGraph.maxSlidersPerPage)))

    const [currentGraphPage, setCurrentGraphPage] = useState(0)
    const lastGraphPage = 1

    const [helpVisible, setHelpVisible] = useState<number | null>(null);

    const [sidebarVisible, setSidebarVisible] = useState(false);

    const [selectedTelescopes,setSelectedTelescopes] = useState<string[]>([]);
    const [redshift, setRedshift] = useState(0);


    // Used for the FixAxes button, in order to zoom into certain elements of the screen. Currently not in use.

    const [fixAxes,setFixAxes] = useState(false)
    const [lastUsedSlider, setLastUsedSlider] = useState(0)

    

    //                                                                     -----     FUNCTIONS     -----

    //                                                          -----     Drawing and rendering all graphs     -----


    function DrawAllCharts(coeffs:number[], fixAxes:boolean, index:number){ 

        const telescopesNoHST = useMemo<string[]>(() => selectedTelescopes.filter(t => t !== "Herschel"),[selectedTelescopes]);

        if (currentGraphPage <= 0){
            return(
                <div className="bg-[#f3f7f5] border border-gray-950 rounded-xl p-6 shadow items-center max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-4">
                    {DrawChart(coeffs,fixAxes,index,"f",selectedTelescopes,redshift)}
                    {DrawChart(coeffs,fixAxes,index,"r",telescopesNoHST,redshift)}
                </div>
            )
        }
        else{
            return(
                <div className="bg-[#f3f7f5] border border-gray-950 rounded-xl p-6 shadow items-center max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-4">
                    {DrawChart(coeffs,fixAxes,index,"n",telescopesNoHST,redshift)}
                    {DrawChart(coeffs,fixAxes,index,"q",telescopesNoHST,redshift)}
                </div>
            )
        }
    }

    function DrawChart(coeffs: number[],fixAxes: boolean,index: number,graph_type: string,telescopes: string[],redshift: number) {

        const [chartData, setChartData] = useState<{ xs: number[]; ys: number[] }>({
            xs: [],
            ys: [],
        });

        const [telescopeData, setTelescopeData] = useState<
            { color: string; xs: number[]; ys: number[] }[]
        >([]);

        // Main data
        useEffect(() => {
            fetch(
            `http://localhost:5000/predict?inputs=${coeffs.join(",")}&graph_type=${graph_type}`
            )
            .then((res) => res.json())
            .then((data) => {
                console.log(data)
                setChartData(data);
            });
        }, [coeffs, graph_type]);

        // Fetch telescope bands
        useEffect(() => {
            fetch(
            `http://localhost:5000/getBands?telescopes=${telescopes.join(",")}&redshift=${redshift}`
            )
            .then((res) => res.json())
            .then((data) => {
                setTelescopeData(data.datasets);
            });
        }, [telescopes, redshift]);

        const [minX, maxX] = useMemo(() => {
            if (!chartData.xs || chartData.xs.length === 0) return [0, 1];
            return [Math.min(...chartData.xs), Math.max(...chartData.xs)];
        }, [chartData.xs]);

        const [minY, maxY] = useMemo(() => {
            if (chartData.ys.length === 0) return [0, 1]; 
            const ysFlat = Array.isArray(chartData.ys[0]) ? chartData.ys.flat() : chartData.ys;
            const min = Math.min(...ysFlat);
            const max = Math.max(...ysFlat);
            return [min, max];
        }, [chartData]);

        const datasets = useMemo(() => {
            const smoothCurve = {
                label: "smooth curve",
                data: chartData.xs.map((x: number, i: number) => ({ x, y: chartData.ys[i] })),
                borderColor: "black",
                backgroundColor: "black",
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 0,
            };

            const telescopeCurves = telescopeData.map((val, idx) => {
                const scaledPoints = val.xs.map((x: number, i: number) => ({
                    x,
                    y: graph_type === "f" ? minY + val.ys[i] * (maxY - minY) : val.ys[i] * (maxY - minY), 
                }));

                const baseline = minY;

                const filledData = [
                    { x: val.xs[0], y: baseline },
                    ...scaledPoints,
                    { x: val.xs[val.xs.length - 1], y: baseline },
                ];

                return {
                    label: `telescope ${idx + 1}`,
                    data: filledData,
                    borderColor: val.color,
                    backgroundColor: val.color + "33", 
                    fill: "origin", // Should fill but doesn't
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                };
            });

            return [smoothCurve, ...telescopeCurves];
        }, [chartData, telescopeData]);

        const data = useMemo(
            () => ({
                datasets: datasets,  
            }),
            [datasets]
        );

        return (
            <div style={{ width: 500, height: 300 }}>
            <Line
                data={data}
                options={{
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                    min: minX,
                    max: maxX,
                    type: "logarithmic",
                    ticks: { display: false },
                    grid: { display: false },
                    border: { color: "#808080" },
                    title: {
                        display: true,
                        text: "Rest Wavelength",
                        color: "#000000",
                        font: { size: 14 },
                    },
                    },
                    y: {
                    type: graph_type === "f" ? "logarithmic" : "linear",
                    min: fixAxes ? spectrumGraph.axesRange[lastUsedSlider][0] : undefined,
                    max: fixAxes ? spectrumGraph.axesRange[lastUsedSlider][1] : undefined,
                    ticks: { display: false },
                    grid: { display: false },
                    border: { color: "#808080" },
                    title: {
                        display: true,
                        text:
                        graph_type === "f"
                            ? "Galaxy Brightness"
                            : graph_type === "r"
                            ? "Galaxy Size"
                            : graph_type === "n"
                            ? "Cuspiness"
                            : graph_type === "q"
                            ? "Projected Shape"
                            : "ERROR",
                        color: "#000000",
                        font: { size: 14 },
                    },
                    },
                },
                }}
            />
            </div>
        );
    }

    //                                                  -----     Handling the slider movement, and adapting the graphs     -----

    function handleSliderChange(sliderIndex: number, value: number, fixAxes:boolean){
        const newValues = [...sliderValues];
        newValues[sliderIndex] = value;
        setSliderValues(newValues);
        setLastUsedSlider(sliderIndex)
        DrawAllCharts(sliderValues,fixAxes,sliderIndex)
    }

    function handleRedshiftSliderChange(value: number, fixAxes: boolean){
        setRedshift(value)
        DrawAllCharts(sliderValues,fixAxes,lastUsedSlider)
    }

//                                                                  -----     Handling Fix Axis Functionality     -----

    function handleFixAxes(checked:boolean, lastSlider: number){
        setFixAxes(checked)
        DrawAllCharts(sliderValues, checked, lastSlider)
    }

    function drawFixAxes(){

        return(
            <div className="flex flex-row items-center justify-center my-2">
                <label className="mr-2 text-white">Fix axes </label>
                <input
                    type="checkbox"
                    checked={fixAxes}
                    onChange={() => handleFixAxes(!fixAxes,lastUsedSlider)}
                />
            </div>
        )
    }



//                              -----     Page based functionality to swtich between menus, for both the sliders and the graphs     -----

    function handlePageChange(currentPage: number, lastPage: number, direction: number, setPage:(value: React.SetStateAction<number>) => void){
        let newPage = currentPage + direction
        if (newPage < 0){
            newPage = lastPage
        }
        if (newPage > lastPage){
            newPage = 0
        }
        setPage(newPage)
        
    }

    function drawPageButtons(noPages:number, button_type:string){

        if (noPages < 1){
            return <div></div>
        }
        else{
            return(
                <div className="flex flex-row items-center justify-center my-4 space-x-4">
                    <button
                        type="button"
                        className="w-16 rounded-e-md bg-slate-900 text-[#e0e1dd] hover:bg-[#604b62]"
                        onClick={button_type === "slider" ? ()=>handlePageChange(currentSliderPage,lastSliderPage,-1,setCurrentSliderPage) 
                            : ()=>handlePageChange(currentGraphPage,lastGraphPage,-1,setCurrentGraphPage)}
                    >
                        ←
                    </button>
                                <button
                        type="button"
                        className="w-16 rounded-e-md  bg-slate-900 text-[#e0e1dd] hover:bg-[#604b62]"
                        onClick={button_type === "slider" ? ()=>handlePageChange(currentSliderPage,lastSliderPage,1,setCurrentSliderPage) 
                            : ()=>handlePageChange(currentGraphPage,lastGraphPage,1,setCurrentGraphPage)}
                    >
                        →
                    </button>
                </div>
            )
        }
    }

    //                                                           -----     Reset button functionality     -----

    function ResetAllValues(){
        setSliderValues(default_values)
        setFixAxes(false)
        setLastUsedSlider(0)
    }

    function drawResetButton(){
        return(
            <div className="flex flex-row items-center justify-center my-4 space-x-4">
                <button
                    type="button"
                    className="w-16 rounded-e-md bg-slate-900 text-[#e0e1dd] hover:bg-[#604b62]"
                    onClick={()=>ResetAllValues()}
                >
                        RESET
                </button>
            </div>

        )
    }

    //                                                        -----     Slider Help Text Functionality     -----

    function displayHelpText(helpVisible:number|null, setHelpVisible: React.Dispatch<React.SetStateAction<number | null>>, currentSliderPage:number){

        if (helpVisible === null) return null;

        const helpText = spectrumGraph.helpTexts[
            currentSliderPage * spectrumGraph.maxSlidersPerPage + helpVisible
        ];

        return (
            <div
            className="fixed inset-0 bg-opacity-60 flex items-center justify-center z-50"
            onClick={() => setHelpVisible(null)}
            >
                <div
                    className="bg-slate-700 rounded-xl px-6 py-4 text-gray-200 text-lg shadow-lg max-w-md text-center"
                    onClick={(e) => e.stopPropagation()} 
                >
                        {helpText}
                        <div className="mt-4">
                            <button
                                className="px-4 py-2 bg-[#5f7baf] text-white rounded hover:bg-[#36465d]"
                                onClick={() => setHelpVisible(null)}
                            >
                                Close
                            </button>
                        </div>
                </div>
            </div>
        )
    }

//                                                         -----     Telescope and Sidebar Functionality     -----

    function drawSidebar(sidebarVisible:boolean, setSidebarVisible:React.Dispatch<React.SetStateAction<boolean>>) {
        
        return (
            <div>

                {/* Overlay (click to close) */}
                {sidebarVisible && (
                    <div
                    className="fixed inset-0 bg-opacity-60 flex items-center z-50"
                    onClick={() => setSidebarVisible(false)}
                    />
                )}

                {/* Sidebar */}
                <div
                    className={`fixed top-0 right-0 h-full w-64 bg-slate-800 text-white p-6 z-50 transform transition-transform duration-300 ${
                    sidebarVisible ? "translate-y-0" : "-translate-y-full"
                    }`}
                >
                    <div className="flex justify-between items-center mb-6">
                        <h2 className="text-xl font-bold"> Telescope Menu</h2>
                        <button className="w-16 rounded-e-md bg-slate-900 text-[#e0e1dd] hover:bg-[#604b62]" onClick={() => setSidebarVisible(false)}>
                            Close
                        </button>
                    </div>
                        <ul className="space-y-4">
                            {spectrumGraph.TelescopeIcons.map((val, idx) => {
                                const telescopeName = spectrumGraph.TelescopeNames[idx];
                                const isSelected = selectedTelescopes.includes(telescopeName);

                                const handleChange = () => {
                                if (isSelected) {
                                    setSelectedTelescopes(selectedTelescopes.filter(t => t !== telescopeName));
                                } else {
                                    setSelectedTelescopes([...selectedTelescopes, telescopeName]);
                                }
                                };

                                return (
                                <div key={idx} className="flex items-center space-x-2">
                                    <input
                                    type="checkbox"
                                    checked={isSelected}
                                    onChange={handleChange}
                                    className="accent-indigo-500"
                                    />
                                    <label>{telescopeName}</label>
                                    <img src={`Telescopes/${val}`} className="w-32 h-32" />
                                </div>
                                );
                            })}
                        </ul>
                        <p className="p-2 text-white text-center"> Red shift: {redshift} </p>
                        <input
                            type="range"
                            min={0}
                            max={4}
                            step={0.5}
                            value={redshift}
                            onChange={e => handleRedshiftSliderChange(Number(e.target.value),fixAxes)}
                        >
                        </input>
                    </div>
                </div>
            );
        }
    
    function drawSidebarButton(){
        return(
            <button
                onClick={() => setSidebarVisible(true)}
                className="fixed top-4 right-4 z-50 p-2 bg-slate-900 text-[#e0e1dd] hover:bg-[#604b62]"
                >
                VIEW TELESCOPES
            </button>  
        )
    }

//                                                          -----     Rendering All Components     -----

    return(
        <div>

            <h1 className = "fixed top-4 left-4 text-5xl font-bold text-[#f7b638] p-6 bg-[#5f7baf] border-gray-950 rounded-xl shadow"> SE3D</h1>

            <img src="RAS_Logo_white.png" className="fixed bottom-4 left-4 w-64 h-64"></img>

            {drawSidebarButton()}

            {drawSidebar(sidebarVisible,setSidebarVisible)}

            {displayHelpText(helpVisible,setHelpVisible,currentSliderPage)}

            <div className="flex flex-col items-center">
                <h1 className="text-center text-2xl font-bold text-white p-6">{spectrumGraph.title}</h1>
                {DrawAllCharts(sliderValues,fixAxes,lastUsedSlider)}
                {drawPageButtons(lastGraphPage,"graph")}
                <p className = "flex items-center justify-center text-white">Page {currentGraphPage+1} of {lastGraphPage+1}</p>
                <p className = "flex items-center justify-center text-white">Click on the icons of the sliders for more information</p>
            </div>

            <div className="bg-[#5f7baf] border border-gray-950 rounded-xl p-6 shadow items-center my-4 max-w-3xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-4"/*className="flex flex-col items-center my-4"*/>
                {slidersOnPage.map((val, idx) => (
                    <div key={idx} className="flex items-center space-x-2 my-2">
                        <img
                            src = {`/SliderIcons/${spectrumGraph.sliderIcons[currentSliderPage * spectrumGraph.maxSlidersPerPage + idx]}`}
                            className="w-16 h-16 mr-2"
                            onClick={() => setHelpVisible(idx)}
                        />
                        <label className="text-white">Coeff {currentSliderPage * spectrumGraph.maxSlidersPerPage + idx + 1}:</label>
                        <input
                            type="range"
                            min={spectrumGraph.sliderRanges[currentSliderPage * spectrumGraph.maxSlidersPerPage + idx][0]}
                            max={spectrumGraph.sliderRanges[currentSliderPage * spectrumGraph.maxSlidersPerPage + idx][1]}
                            step={(spectrumGraph.sliderRanges[currentSliderPage * spectrumGraph.maxSlidersPerPage + idx][1] -
                                    spectrumGraph.sliderRanges[currentSliderPage * spectrumGraph.maxSlidersPerPage + idx][0]) / 10}
                            value={val}
                            onChange={e => handleSliderChange(currentSliderPage * spectrumGraph.maxSlidersPerPage + idx, Number(e.target.value), fixAxes)}
                            className="custom-slider"
                        />
                        <span className="w-10 text-center inline-block text-white">{val}</span>
                    </div>
                ))}
            </div>

            {drawFixAxes()}

            {drawPageButtons(lastSliderPage,"slider")}

            {drawResetButton()}
            
            <p className = "flex items-center justify-center text-white">Page {currentSliderPage+1} of {lastSliderPage+1}</p>
        </div>
    )
}


/*
            <div className="flex flex-row items-center justify-center my-2">
                <label className="mr-2 text-white">Fix axes </label>
                <input
                    type="checkbox"
                    checked={fixAxes}
                    onChange={() => handleFixAxes(!fixAxes,lastUsedSlider)}
                />
            </div>
*/
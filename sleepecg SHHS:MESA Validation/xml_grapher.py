import os
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

def graph_xml(xml_file, epoch_length=30):
    """
    Parses an SHHS XML file to extract sleep stage annotations.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    scored_events = root.findall("ScoredEvents/ScoredEvent")
    sleep_events = [scored_event for scored_event in scored_events if scored_event.find("EventType").text == "Stages|Stages"]

    sleep_stages = []
    
    start = 0
    for event in sleep_events:
        duration = float(event.find("Duration").text)
        sleep_stage = event.find("EventConcept").text[:-2]
        if sleep_stage not in ["Wake", "REM sleep"]:
            sleep_stage = "Non-REM"
        num_sub_events = int(duration/epoch_length)
        for i in range(num_sub_events):
            end = start + epoch_length
            sleep_stages.append([start,epoch_length, end,sleep_stage])
            start = end

    df = pd.DataFrame(sleep_stages, columns=["Start_Time", "Duration", "End", "Sleep_Stage"])

    return df






if __name__ == "__main__":
    input_file = "shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml"


    df = graph_xml(input_file)

    # Plot sleep stages as a line plot
    plt.figure(figsize=(10, 5))
    plt.step(df["Start_Time"], df["Sleep_Stage"], linewidth=2)

    # Formatting
    plt.yticks([0, 1, 2], ["Wake", "REM sleep", "Non-REM"])  # Label sleep stages
    plt.gca().invert_yaxis()  # Invert y-axis so Wake is at the top
    plt.xlabel("Time (seconds)")
    plt.ylabel("Sleep Stage")
    plt.title(f"Sleep Hypnogram")
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    # Show plot
    plt.show()




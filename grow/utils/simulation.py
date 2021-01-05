from lxml import etree


def write_configs_to_base(
    file_path, elastic_mod, density, simulation_time
):
    doc = etree.parse(file_path)

    doc.xpath("/VXA/VXC/Palette/Material/Mechanical/Elastic_Mod")[0].text = str(
        elastic_mod
    )
    doc.xpath("/VXA/VXC/Palette/Material/Mechanical/Density")[0].text = str(density)
    doc.xpath(
        "/VXA/Simulator/StopCondition/StopConditionFormula/mtSUB/mtCONST"
    )[0].text = str(simulation_time)

    file_content = etree.tostring(doc, pretty_print=True).decode("utf-8")
    with open(file_path, "w") as f:
        print(file_content, file=f)

from lxml import etree


def get_voxel_positions(out_file_path):
    doc = etree.parse(out_file_path)

    def parse(x):
        y = x.split(";")
        p = []
        for v in y:
            if len(v) > 0:
                p.append([float(q) for q in v.split(",")])
        return p
    
    intitial_positions = doc.xpath("/report/detail/robot/init_pos")[0].text
    final_positions = doc.xpath("/report/detail/robot/pos")[0].text
    return parse(intitial_positions), parse(final_positions)


def get_fitness(out_file_path):
    """Get the best fitness from the simulation output.

    """

    doc = etree.parse(out_file_path)
    return float(doc.xpath("//fitness_score")[0].text)

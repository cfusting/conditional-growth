from lxml import etree


def get_voxel_positions(out_file_path, voxel_size=0.01):
    doc = etree.parse(out_file_path)

    def parse(x):
        y = x.split(";")
        p = []
        for v in y:
            if len(v) > 0:
                p.append([float(q) / voxel_size for q in v.split(",")])
        return p
    
    initial_positions = doc.xpath("/report/detail/robot/init_pos")[0].text
    final_positions = doc.xpath("/report/detail/robot/pos")[0].text
    return parse(initial_positions), parse(final_positions)

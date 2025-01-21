import os

def insert_tab(tab_name,destination_content,
               destination_type=None,
               output=None):

    source = tab_name + '.tex'

    with open(source, 'r') as source:
        source_content = source.read()

    if destination_type is not None:
        destination = destination_content + destination_type

        with open(destination, 'r') as destination:
            destination_content = destination.read()

    insert_location = r'% INSERT ' + tab_name

    # Find the insertion point in the destination content
    insert_index = destination_content.find(insert_location)

    # Insert the source content at the specified location
    updated_destination = (
            destination_content[:insert_index]
            + source_content
            + destination_content[insert_index:]
        )
    
    if output is None:
        return updated_destination
    else:
        with open(output, 'w') as f:
            f.write(updated_destination)


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    insert_tab(tab_name='psm_risk_tab',
                destination_content='psm_original',
                destination_type='.txt',
                output='psm_reg.tex')
    
    insert_tab(tab_name='psm_time_tab',
               destination_content='psm_reg',
               destination_type='.tex',
               output='psm_reg.tex')
    
    
    






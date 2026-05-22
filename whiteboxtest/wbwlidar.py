import whitebox_workflows as wbw
from whitebox_workflows import WbEnvironment

wbe = WbEnvironment()

# Let's begin by downloading the Whitebox Workflows 'Kitchener_lidar' sample data
wbe.working_directory = wbw.download_sample_data('Kitchener_lidar')
print(f'Data have been stored in: {wbe.working_directory}')

# Read in an existing lidar data set
lidar = wbe.read_lidar('Kitchener_lidar.laz')

# To create a new Lidar object, you need a LidarHeader, which documents metadata
# about a LiDAR file. It is the LiDAR equivalent to the RasterConfigs.
print(f'File creation day: {lidar.header.file_creation_day}')
print(f'File creation year: {lidar.header.file_creation_day}')
print(f'Generating software: {lidar.header.generating_software}')
num_points = lidar.header.get_num_points()
print(f'Number of points: {num_points}')
print(f'Version major: {lidar.header.version_major}')
print(f'Version minor: {lidar.header.version_minor}')
print(f'Point format: {lidar.header.point_format}')
print(f'Min X: {lidar.header.min_x}')
print(f'Max X: {lidar.header.max_x}')
print(f'Min Y: {lidar.header.min_y}')
print(f'Max Y: {lidar.header.max_y}')
print(f'Min Z: {lidar.header.min_z}')
print(f'Max Z: {lidar.header.max_z}')

# Now, create a new Lidar object. In doing so, the new Lidar object will copy 
# over some of the properties in the source LidarHeader, but won't copy any of 
# the things like generating software, creation day/year, point extent values,
# or the source point data. It's a newly initialized file ready to receive its
# own point data.
lidar_out = wbe.new_lidar(lidar.header)

# You likely want to copy over the VariableLengthRecord (VLR) data too. VLRs
# usually contain important information, such as the coordinate reference 
# system.
lidar_out.vlr_data = lidar.vlr_data

print('Filtering point data...')
old_progress = -1
for i in range(num_points):
    # Notice that if the file does not contain time, colour, or waveform data,
    # each of these will simply be None. You can use the has_time_data(),
    # has_colour_data(), and has_waveform_data() methods to determine if these
    # data are stored in the file.
    point_data, time, colour, waveform = lidar.get_point_record(i)

    # The PointData returned by the get_point_record method has the raw
    # untransformed point coordinate information as well as all the info
    # about point intensity, class, return values, etc. If you simply want
    # the transformed x,y,z coordinates, use the get_transformed_xyz method
    # instead.
    
    # Now let's filter the data based on return data...
    if point_data.is_first_return() or point_data.is_intermediate_return():
        # Save the point to lidar_out
        lidar_out.add_point(point_data, time, colour, waveform)

    # Update the progress once we've completed another 1% of the points.
    progress = int((i + 1.0) / num_points * 100.0)
    if progress != old_progress:
        old_progress = progress
        print(f'Progress: {progress}%')


# Write lidar_out to file
wbe.write_lidar(lidar_out, "new_lidar.laz")
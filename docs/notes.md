# TODO

- find way to display shapes (preferably interactively)
- CI/CD


## Prototype
1. Start with a square
2. Extrude square into cube
3. Create a POI
4. Calculate visibility
5. Display cube faces

## Python packages/etc
- geometry: shapely, fiona, geopandas, cuspatial?
- pydantic, pint
- test: pytest, factoryboy
- infra: dask, poetry, xarray

## Reference
### Acceleration structures (fewer ray-object intersection tests)
- Bounding Volume Hierarchies
  - https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
  - https://fileadmin.cs.lth.se/cs/Education/EDAN30/lectures/S2-bvh.pdf
  - https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-acceleration-structure/introduction
  - https://my.eng.utah.edu/~cs6965/slides/05-josef.pdf

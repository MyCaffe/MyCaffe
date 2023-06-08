# start with a base image with all the necessary tooling to compile our app
FROM mcr.microsoft.com/dotnet/framework/sdk:4.8 AS runtime
CMD [ "cmd" ]

# Copy mycaffe.app.setup.exe to container
COPY MyCaffe.app.setup.exe C:\mycaffe.app.setup.exe

# Install mycaffe.app.setup.exe
RUN C:\MyCaffe.app.setup.exe /q



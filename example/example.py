import pandavro as pdx


def main():
    weather = pdx.from_avro('weather.avro')

    print(weather)

    pdx.to_avro('weather_out.avro', weather)

if __name__ == '__main__':
    main()
